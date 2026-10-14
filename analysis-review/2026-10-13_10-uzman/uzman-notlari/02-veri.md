# Uzman 02 — Veri kalitesi, ayrımlar ve sızıntı

İnceleme tarihi: 13 Ekim 2026. Bu, ayrı bir yapay zekâ uzman incelemesidir; uygulama düzeltmesi yapılmadı.

## Sonuç ve sınırlar

En önemli doğrulanmış sorun, her model için yeniden rastgele train/validation/test ayrımı oluşturulmasıdır. Modeller aynı test dosyalarında karşılaştırılmayabilir. Bununla birlikte anomalib 2.6.0 `Folder` varsayılanı `FROM_TEST` olduğundan **aynı çalıştırma içinde validation ve testin aynı dosyalardan oluştuğu iddiası doğru değildir**. Metadata deneyinde train–validation, train–test ve validation–test dosya kesişimleri boş çıktı. Aynı nesne Lightning tarafından kurulurken `_is_setup` bayrağı da ayrımın korunmasını sağlar (`.venv/Lib/site-packages/anomalib/data/datamodules/base/image.py:145–164`).

Proje envanterindeki 230 eğitim iş JSON'u okundu: 131 gerçek iş, 99 demo iş. Gerçek işlerin 119'unda `mask_dir` boş, 12'sinde `masks`; bu bir kusur veya başarı oranı değildir, piksel düzeyindeki doğruluğu sınamak için maske kanıtının çoğu kayıt için bulunmadığını gösterir. Bu işlerde geçen sekiz ayrı Desktop dataset yolu inceleme sırasında mevcut değildi. Dolayısıyla gerçek verinin etiket doğruluğu, görüntü/maskelerin boyutları, içerik tekrarları, sınıf sayıları veya gerçek sızıntı oranı doğrulanamadı. `.data/camera-smoke-test` altında tek normal PNG kayıtlı. Başlıca iş yolları ve sayımları [veri iş kaydı özeti](../kanitlar/veri-is-kayit-ozeti.json) içinde saklandı.

Verilen build/veri envanterinde `metrics.csv` ve `hparams.yaml` bulunmadı. Mevcut iş JSON'ları iş niyetini gösterir; işlerin başarıyla tamamlandığını veya bugün incelenen kaynak sürümüyle yürütüldüğünü kanıtlamaz. Örnek son gerçek iş: [train-20260811-163457-810.json](../../../.data/migrated-builds/Desktop_Qt_6_11_1_MSVC2022_64bit_Release/release/jobs/train-20260811-163457-810.json).

## Bulgular

### V02-01 — Model karşılaştırmasında değerlendirme dosyaları sabit değil

- **Tür / önem:** Doğrulanmış hata; P1.
- **Kanıt:** [create_datamodule](../../../worker/anomalib_worker.py) `seed`, `test_split_ratio` veya `val_split_mode` geçirmiyor. [model döngüsü](../../../worker/anomalib_worker.py) her model için yeni datamodule kuruyor. [kazanan seçimi](../../../worker/anomalib_worker.py) bu sonuçların test AUROC değerlerini karşılaştırıyor. Kurulu 2.6.0 `Folder` için `seed=None`, normal test oranı 0.2 ve validation/test havuzu ayrımı 0.5: `.venv/Lib/site-packages/anomalib/data/datamodules/image/folder.py:121–136`; rastgele permütasyon: `data/utils/split.py:176–179`.
- **Çalışan kanıt:** 20 normal + 8 anomalili sentetik dosya metadata havuzu ile art arda iki `worker.create_datamodule(...).setup()` çalıştırıldı. Her ikisinde sayılar 16 train / 6 validation / 6 test olsa da dosyalar farklıydı; `seed=42` ile iki ayrım aynıydı. [Ölçüm kaydı](../kanitlar/veri-metadata-kontrol.json).
- **Tetikleyici:** Aynı eğitim işinde PatchCore ve PaDiM gibi iki modeli seçmek veya aynı parametreleri yeniden denemek.
- **Etki:** Skor farkı modelden veya seçilen örneklerin zorluğundan kaynaklanabilir; kazanan kararı ve tekrar üretilebilirlik zayıflar. Bu deney bir gerçek model başarımı ölçümü değildir.
- **Çözüm:** Veri ayrımını model döngüsünden önce üretip dosya manifesti ve sabit, pozitif ayrım tohumu kaydetmek; tüm modeller aynı ayrımı kullanmalı. Sonuçlara manifest kimliği, sınıf/ayrım sayıları ve içerik/dataset sürümü eklenmeli. Üretim kararını ayrıca model seçimine hiç katılmamış holdout üzerinde sınamak gerekir.
- **Kabul testi:** İki model ve iki yeniden çalıştırma aynı değişmeyen dataset manifestinde aynı train/validation/test dosya kümelerini kullanır; kesişimler boştur; bir veri dosyası değişince manifest kimliği değişir ve eski sonuçla aynı datasetmiş gibi karşılaştırılmaz.
- **Efor:** 1–2 gün; grup temelli ayrım ayrıca V02-04 kapsamındadır.

### V02-02 — Yetersiz veya yalnız normal veri eğitim öncesinde açıklanabilir biçimde engellenmiyor

- **Tür / önem:** Doğrulanmış hata (sentetik metadata reproduksiyonu); P2.
- **Kanıt:** [gerçek eğitim kontrolü](../../../mainwindow.cpp) yalnız dataset ve normal klasörü varlığını kontrol ediyor; görüntü sayısı, anomalili sınıf ve ortaya çıkacak validation/test sınıf sayıları doğrulanmıyor. [preflight_selected_models](../../../worker/anomalib_worker.py) model varlıkları/parametrelerine odaklı. anomalib `random_split` boş alt kümeyi yalnız uyarır (`.venv/Lib/site-packages/anomalib/data/utils/split.py:166–173`).
- **Çalışan kanıt:** Tek normal dosya ve anomali klasörü olmayan sentetik veri `Folder.setup()` içinde `ValueError: not enough values to unpack (expected 2, got 0)` üretti. [Hata kaydı](../kanitlar/veri-metadata-kontrol.json). Gerçek kamera smoke klasörünün de yalnız bir normal dosyası var; envanterde bu klasöre ait iş **demo** olduğu için geçmiş gerçek eğitim başarısızlığı iddia edilmiyor.
- **Tetikleyici:** Boş normal klasörü, yalnız bir normal görüntü, tek anomalili görüntü veya normal-only datasetle “real” eğitimi başlatmak. Anomalili sayısı bir olduğunda 0.5 ayrımı bir tarafı anomalisiz bırakabilir.
- **Etki:** Operatör veri sorununu genel Python hatası olarak görür; daha büyük normal-only veriyle sınıflandırma AUROC/F1 değerlendirmesi iki sınıflı kanıt sağlamaz. Normal-only model öğrenimi ile iki sınıflı model karşılaştırmasının amaçları ayrılmalıdır.
- **Çözüm:** Model kurulumundan önce decode edilebilir örnekler ve planlanan ayrım sınıf sayıları için dataset preflight; geçersiz karşılaştırma için açık minimumlar/hata. İstenirse normal-only öğrenme ayrı seçenek olarak desteklenmeli ve iki sınıflı metrikler “hesaplanamaz” gösterilmeli.
- **Kabul testi:** Boş klasör, 1 normal/0 anomali ve yeterli normal/1 anomali örnekleri model kurulmadan anlaşılır veri hatası üretir; geçerli kümede validation ve testin her ikisinde sınıflar bulunur. Normal-only öğrenme seçeneği varsa başarılı öğrenme sonucu ile metrik kullanılabilirliği açıkça ayrılır.
- **Efor:** 0.5–1 gün.

### V02-03 — Belirsiz maske adı geri dönüşü yanlış ground truth gösteriyor

- **Tür / önem:** Doğrulanmış hata (sentetik dosya eşleme reproduksiyonu); P2.
- **Kanıt:** Worker [matching_mask](../../../worker/anomalib_worker.py), göreli yol bulunmayınca bütün maske ağacındaki ilk eşleşen stem'i seçiyor; UI [matchingMaskForImage](../../../mainwindow.cpp) aynı yaklaşımı uyguluyor. Bu geri dönüş alt klasör kimliğini kaybediyor ve çoklu eşleşmeyi hata saymıyor. Eğitim yolu bu fonksiyonu kullanmaz; doğrudan anomalib'e gider. Kurulu `Folder` ayrıca sıralı maske listesi ve stem içerme kontrolü kullanır (`.venv/Lib/site-packages/anomalib/data/datasets/image/folder.py:245–262`).
- **Çalışan kanıt:** `abnormal/b/000.png` için doğru göreli maske yokken `masks/a/000_mask.png` ve `masks/c/000_mask.png` arasında ilk maske seçildi. [Eşleme kaydı](../kanitlar/veri-metadata-kontrol.json).
- **Tetikleyici:** Ayrı ürün/hata alt klasörlerinde tekrar kullanılan `000.png` gibi adlar ve eksik/yanlış yerleştirilmiş maske.
- **Etki:** Operatörün gördüğü gerçek maske başka görüntüye ait olabilir; görsel değerlendirme ve hata teşhisi yanıltılır. Bu bulgu gerçek eğitim maskelerinin karıştığını kanıtlamaz; preview eşlemesindeki yanlışlığı kanıtlar.
- **Çözüm:** Tam göreli yol/stem eşlemesini ortak bir manifestte tanımlamak; global geri dönüş yalnız benzersiz aday varsa uygulanmalı. Hiç aday/çok aday, eksik fazla maskeler ve görüntü–maske boyut farkları preflight'ta açıkça raporlanmalı; UI ve eğitim aynı eşleme sözleşmesini kullanmalı.
- **Kabul testi:** İki alt klasörde aynı stem ile doğru maskeler doğru görüntülere bağlanır; eksik göreli maskeye karşı iki global aday varsa maske gösterilmez ve belirsizlik bildirilir; tüm eğitim eşlemeleri UI eşlemesiyle aynıdır.
- **Efor:** 0.5–1 gün; ortak manifest uygulaması V02-01 ile birleşebilir.

### V02-04 — Dosya bazlı rastgele ayrım, aynı parçanın/çekim serisinin bağımlı görüntülerini ayırabilir

- **Tür / önem:** Koşullu risk; P1 (model başarımının üretime genellenmesi hedefleniyorsa).
- **Kanıt:** Kamera toplama [nextCapturePath](../../../mainwindow.cpp) yalnız sınıf + zaman damgalı dosya adı üretiyor; parça/lot/oturum grup kimliği yok. Worker [create_datamodule](../../../worker/anomalib_worker.py) grup bilgisi veya bağımsız normal test yolu geçirmiyor. `Folder` tüm `normal` alt ağacını train adayı sayar (`.venv/Lib/site-packages/anomalib/data/utils/path.py:116`, `data/datasets/image/folder.py:279–286`), ardından dosya bazlı rastgele böler.
- **Tetikleyici:** Aynı fiziksel parçanın ardışık kareleri, aynı çekimden üretilen augmentasyon/crop'lar, byte-identical kopyalar veya aynı lot/ışık koşulunun train ve testte dağılması. `normal/train` ve `normal/test` gibi kullanıcı ayrımları da mevcut sözleşmede özel split olarak yorumlanmaz; README'nin desteklediği yapı düz normal/abnormal havuzudur.
- **Etki:** Dosya yolu kesişimi sıfır olsa bile bağımlı örnekler değerlendirmeyi kolaylaştırabilir. Gerçek datasetler mevcut olmadığından bu projede sızıntı gerçekleştiği veya ne kadar gerçekleştiği **tespit edilmedi**. False alarm ve kaçırma oranı uydurulmadı.
- **Çözüm:** Parça/lot/oturum metadata'sı, tam içerik hash'i ve gerektiğinde yakın-kopya taraması; ayrımı grup düzeyinde yapmak. Son üretim doğrulamasını ayrı zaman/lot/kamera koşulları üzerinde kurmak ve bağımsız normal test yolunu desteklemek.
- **Kabul testi:** Aynı grup kimliğine veya içerik hash'ine ait tüm görüntüler tek splitte kalır; rapor grup/hash çakışmasını sıfır doğrular; final holdout model/eşik seçimi ve augmentasyon kaynağı olarak kullanılmaz.
- **Efor:** 1–3 gün yazılım/manifest; mevcut veri için etiketleme ve yeniden toplama süresi veri erişimine bağlıdır.

### V02-05 — UI/worker sayımı ile gerçek eğitimdeki dosya kapsamı farklı

- **Tür / önem:** Doğrulanmış hata; P2.
- **Kanıt:** [UI uzantı filtreleri](../../../mainwindow.cpp) ve worker [IMAGE_EXTENSIONS](../../../worker/anomalib_worker.py) PNG/JPEG/BMP/TIF ile sınırlı. [create_datamodule](../../../worker/anomalib_worker.py) `extensions` geçirmediğinden anomalib torchvision'un daha geniş listesini kullanır: `.venv/Lib/site-packages/torchvision/datasets/folder.py:257` ayrıca WEBP/PPM/PGM içerir.
- **Çalışan kanıt:** Aynı normal klasöründe bir `.png` ve bir `.webp` metadata dosyasıyla worker sayımı 1, `Folder` train adayı sayımı 2 oldu. [Sayım kaydı](../kanitlar/veri-metadata-kontrol.json). UI'nın kaynak filtreleri worker ile aynı olduğundan UI etkisi statik olarak çıkarılmıştır; Qt GUI çalıştırılmadı.
- **Tetikleyici:** Desteklenen klasöre WEBP/PPM/PGM eklemek veya arşivde bu tür görünmeyen dosyalar bulunması.
- **Etki:** Operatörün gördüğü örnek sayıları ve preview dosyaları gerçek modelin kullandığı kapsamı tam yansıtmaz; görünmeyen dosyalar veri ayrımına girer.
- **Çözüm:** Desteklenen uzantıları tek sözleşmede toplayıp `Folder(extensions=...)` içine açıkça geçirmek veya UI/worker'ı anomalib'in tüm desteklenen uzantılarıyla hizalamak; decode edilip kullanılabilen dosyaların manifest sayısını göstermek.
- **Kabul testi:** PNG/JPEG/BMP/TIF/WEBP/PPM/PGM karışık kümede UI/worker/Folder dosya listeleri aynıdır; destek dışı/bozuk dosyaların neden dışlandığı görünürdür.
- **Efor:** 0.25–0.5 gün.

## Gerçekten yürütülen kontroller

1. Kaynak ve kurulu anomalib 2.6.0 paketinin Folder/dataset/split/path kodu statik okundu; sürüm `.venv/Lib/site-packages/anomalib-2.6.0.dist-info/METADATA:3` ve çalışma zamanı sürümüyle doğrulandı.
2. 230 küçük train iş JSON'u okundu ve sayımları kaydedildi; model/checkpoint içerikleri açılmadı. Dataset yolları yalnız varlık kontrolüne tabi tutuldu.
3. `.venv/Scripts/python.exe -B -X utf8 analysis-review/2026-10-13_10-uzman/kanitlar/veri-metadata-kontrol.py` çalıştırıldı; `PYTHONDONTWRITEBYTECODE=1` ve worker socket network guard etkin. Bu **yeni analiz reproduksiyonudur**, mevcut ürün testinin geçtiği anlamına gelmez. Çıkış kodu 0; beklenen tiny-dataset hata sonucu JSON içinde yakalandı.
4. Sentetik fixture dosyaları gerçek görüntü değildir; yalnız dosya metadata'sı için yaratıldı ve **hiç decode edilmedi**. Eğitim, inference, model yükleme, GPU iş yükü, kamera, install/download/upload yok. Üretim datasetine ve kaynağa yazılmadı; yeni dosyalar yalnız inceleme klasörüne yazıldı.

[Analiz reproduksiyon betiği](../kanitlar/veri-metadata-kontrol.py), [çıktı JSON'u](../kanitlar/veri-metadata-kontrol.json) ve `kanitlar/rol02-sentetik-veri` fixture ağacı saklandı.

## Resmi sürüm/API doğrulaması

Kurulu sürümün varsayılanları resmi projenin **lib/v2.6.0** etiketindeki API docstring ve imza ile eşleştirildi: [Folder API ve dokümante edilmiş varsayılanlar](https://github.com/open-edge-platform/anomalib/blob/lib/v2.6.0/src/anomalib/data/datamodules/image/folder.py#L115), [base image ayrım uygulaması](https://github.com/open-edge-platform/anomalib/blob/lib/v2.6.0/src/anomalib/data/datamodules/base/image.py#L297). Genel resmi [v2.1.0 API dokümantasyonu](https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/data/datamodules/image.html) da aynı Folder ayrım kavramlarını açıklar, ancak sürüme özel kararlar bunun yerine 2.6.0 kurulu kaynak ve resmi etiket üzerinden verildi. 2.6.0 ReadTheDocs URL'leri web aracıyla erişilemedi; resmi sürüm kaynağının docstring'i kullanıldı.

Önerilen sıra: V02-01 ayrım/manifest → V02-02 veri preflight → V02-03 ortak maske eşlemesi ve V02-05 uzantı hizalama → gerçek veri erişimiyle V02-04 grup bağımsızlığı ve holdout doğrulaması.
