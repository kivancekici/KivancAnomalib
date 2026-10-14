# KivancAnomaLibTest — 10 uzmanlı teknik inceleme

13 Ekim 2026 · Europe/Istanbul · Yalnız analiz ve öneri

**10 ayrı yapay zekâ uzmanının tamamı incelemesini bitirdi.** Mevcut 37 worker testi geçti. Buna rağmen üretime seçilecek sonucun kökeni, exportların üzerine yazılması, piksel haritasının kalibrasyonu ve modellerin ortak veri ayrımı üzerinde karşılaştırılması için somut kusurlar bulundu. Bunlar giderilmeden test başarısı tek başına üretim kabulü olarak kullanılmamalı.

10 uzman notundaki **53 bulgu, tekrarlar ve birlikte uygulanabilecek değişiklikler birleştirilerek 43 numaralı iş kalemine** dönüştürüldü. Birleştirilen farklı hata nedenleri her kayıtta ayrı belirtildi. Doğrulanmış kusur, koşullu risk ve iyileştirme fırsatı ayrımı korunmuştur. Bu rapordaki çözümler uygulanmadı; kaynak veya konfigürasyon değiştirilmedi.

## İlk yapılacak 9 iyileştirme

| Sıra | Somut sonuç | İlgili kayıtlar | İlk kabul kapısı |
| --- | --- | --- | --- |
| 1 | Üretim manifestini yalnız tamamlanmış gerçek run ve o run'ın dataset/artifact kimliğinden üretmek | B01, B24, B25 | Demo/eksik checkpoint reddedilir; A sonucu B dataset köküyle kaydedilemez |
| 2 | Export ve managed varlık yenilemede önceki geçerli dosyaları korumak | B02, B07, B22 | İkinci export ve enjekte edilmiş rename/validation hatası ilk geçerli kopyayı bozmaz |
| 3 | Karar için kalibre edilmiş float harita ve sonlu skor kullanmak | B03, B04, B08, B09 | Düşük/yüksek/sabit haritalar doğru maske verir; eksik/NaN skor karar üretmez |
| 4 | Bütün modelleri aynı split manifestiyle değerlendirmek; bağımsız son kabul verisi ayırmak | B05, B06, B14–B16 | Aynı karşılaştırmanın dosya/grup kimlikleri sabittir; holdout ayrımı doğrulanır |
| 5 | Eğitimden inference'a decode/resize/dtype/input/output sözleşmesini tanımlamak | B10–B13, B23 | Küçük bağımsız fixtures ile sayısal parity; geçersiz graph load aşamasında reddedilir |
| 6 | Model yükleme, demo geçişi, sonuç gösterimi ve capture hedefini tek durum sözleşmesine bağlamak | B26–B29, B37, B38 | Eski request/mode/input sonucu yeni seçime aitmiş gibi gösterilemez |
| 7 | Model parametrelerini kurulu sürümün gerçek kısıtlarına göre doğrulamak | B17–B21 | Geçersiz kanal/feature/coreset erken reddedilir; scheduler planlanan adımda değişir |
| 8 | Gecikme, gerçek provider ve bellek bütçesini ölçülebilir hale getirmek | B30–B36 | Keşif/yürütme ayrılır; tutarlı p50/p95 ölçümü ve kapasite planı oluşur |
| 9 | Test edilen worker'ı dağıtım paketine ve sürüm kilidine bağlamak | B39–B43 | Paket SHA/handshake ile test kimliği eşleşir; kritik outcome regresyonları geçer |

Eforlar aşağıda geliştirici günü olarak verilmiştir. Bunlar tek tek yaklaşık tahminlerdir; ortak değişiklikler nedeniyle toplanarak proje takvimi çıkarılmamalı. Gerçek veri toplama, yeniden eğitim ve donanım kabulü ayrıca planlanmalıdır.

## Mevcut durum ve inceleme sınırı

Uygulama Qt Widgets/C++17/qmake arayüzü, QProcess üzerinden JSON Lines konuşan Python worker ve isteğe bağlı Basler Pylon/OpenCV kamera hattından oluşuyor. Ana pencere 4.190, worker 3.487 satır. Worker eğitim, export, varlık hazırlama ve kalıcı inference oturumunu aynı dosyada yönetiyor. Git deposu ve uygulanabilir AGENTS.md bulunmadı; Git işlemi yapılmadı.

Çalışan ortam Python 3.12.14, Anomalib 2.6.0, Torch 2.13.0+cu130, TorchVision 0.28.0+cu130, Lightning 2.6.5, OpenVINO 2026.2.1, ONNX 1.22.0 ve ORT GPU 1.28.0. Paket metadata kontrolü temiz. Qt qmake 6.11.1; Pylon başlıkları 12.2.0.1265 ve native OpenCV 4.12.0 mevcut. Tam kurulum sürüm kilidi yok.

Başlangıç envanterinde `.data` altında 11, `build` altında 2.539 dosya vardı. Tek ONNX teslim dosyası 607.008.673 bayt; bu büyük model açılmadı/yüklenmedi. Küçük sidecar manifestinin `float16`, boyut ve metrik beyanları graph dtype'ının veya yeni kalite ölçümünün kanıtı değildir. Proje envanterinde `.ckpt/.pt` yok. İş kayıtlarının işaret ettiği sekiz harici Desktop dataset yolu erişilebilir değildi. 230 eğitim iş JSON'unda 131 gerçek, 99 demo konfigürasyonu bulundu; bunlar 230 tamamlanmış bağımsız eğitim anlamına gelmez. Gerçek işlerin 119'unda mask_dir boş olması da hata oranı değildir.

İnceleme sırasında native GUI/build, kamera, gerçek üretim modeli, eğitim, indirme, kurulum veya upload çalıştırılmadı. Uzmanların yeni küçük fixtures, AST/metadata kontrolleri, fake backend'leri ve yalnız CPU'da birkaç küçük ONNX graph'ı ayrı kanıt olarak kullanıldı. Gerçek AUROC/F1, yanlış alarm/kaçırma oranı, FPS, GPU gecikmesi ve üretim modelinin backend parity'si ölçülmedi. README/design-qa'daki eski native/kamera test beyanları bu incelemede yeniden yapılmış test sayılmadı.

Olumlu kontroller de var: bozuk JSON/bilinmeyen komut servis döngüsünü durdurmuyor; başarısız yeni model adayında önceki backend korunuyor; asset-lock tek dosya olarak geçici dosyadan replace ile yazılıyor; worker binary mask resize'ında nearest-neighbor kullanıyor; kamera RGB/BGR ve boyut sırası için bu kapsamda bir hata saptanmadı. Kaynakta QtConcurrent/QFuture tabanlı kamera görevi yok; doğrulanmış use-after-free veya kalıcı orphan süreç bulgusu yok.

## Gerçekte çalıştırılan doğrulamalar

| Kontrol | Sonuç | Ne kanıtlar / sınırı |
| --- | --- | --- |
| Mevcut unittest suite, review içi TEMP/TMP | **37/37 OK**, exit 0; unittest 67,930 s, toplam süreç 70,347 s | Kapsanan worker davranışları; gerçek eğitim veya native UI kabulü değildir |
| İlk varsayılan TEMP denemesi | 37 test, 29 PermissionError | Sandbox geçici dizin erişimi; review TEMP ile giderildi, proje hatası sayılmadı |
| Gerçek worker run_check | ok=true; RTX 3060 Laptop GPU, ORT TensorRT/CUDA/CPU, OV CPU/iGPU/dGPU keşfi | Provider/cihaz keşfi; gerçek GPU graph/model yürütmesi değildir |
| pip check | exit 0, No broken requirements found | Paket metadata bağımlılıkları; DLL/driver/model doğrulaması değildir |
| Uzman 02 metadata fixtures | Sabitlenmeyen split, küçük dataset, belirsiz preview maskesi ve kapsam farkı yeniden üretildi | Gerçek veri decode/eğitimi yapılmadı; tek split içi train/val/test kesişimi boştu |
| Uzman 03 AST/teacher metadata/scheduler | Kanal/feature/kısıt sınırları ve scheduler aralığı doğrulandı | Teacher ZIP pickle opcode metadata okundu; unpickle/torch.load/model forward yapılmadı |
| Uzman 04 küçük sayısal fixtures | Kalibrasyon kaybı, geçersiz skor/JSON ve winner seçim problemi yeniden üretildi | Gerçek üretim modeli skoru değildir |
| Uzman 05 küçük görüntü/fake backend fixtures | Uint8 0/1 ölçek kusuru, resize tensor farkı ve 16-bit kırpılma doğrulandı | Girdi tensor farkı kalite kaybının büyüklüğünü ölçmez |
| Uzman 06 tiny ONNX CPU + fake exporter | Export çakışması, raw-map değişimi, external-data ve graph sözleşmeleri doğrulandı | CPUExecutionProvider; 607 MB üretim modeli/GPU kullanılmadı |
| Uzman 07 fake clock/provider/weakref | Timer kapsamı, stale provider etiketi ve çift backend ömrü doğrulandı | 20/5 ms sentetik clock örneği; gerçek benchmark/OOM değildir |
| Uzman 08 hata enjeksiyonu | Eski managed klasörün kaybı ve eksik output_root kabulü yeniden üretildi; servis izolasyonu geçti | Yalnız yeni review fixtures; gerçek disk/kamera/GUI arızası oluşturulmadı |
| Uzman 09 demo check | Anomalib/Torch import girişimi olmadan ok=true | Demo check'in gerçek ortam kontrolü olmadığı doğrulandı |
| Uzman 10 resource hash | 4 üretilmiş resource worker'ının 3'ü kaynakla farklı | Generated C++ resource okundu; mevcut EXE'nin linked/embedded içeriği doğrulanmadı |
| Son koruma kontrolü | 25 kaynak dosyası SHA/boyut/mtime aynı; 2.550 veri/build dosyası boyut/mtime aynı; ek dosya yok | Büyük ikililerin içeriği hash'lenmedi; metadata kontrolünün sınırı korunmalı |

Ana kayıtlar: [envanter](ENVANTER.md), [test logu](kanitlar/unittest-workspace-temp.log), [test komutu/sonucu](kanitlar/unittest-workspace-temp-sonuc.json), [ortam kontrolü](kanitlar/ortam-check.log), [pip check](kanitlar/10-pip-check.log), [son baseline kontrolü](kanitlar/baseline-son-kontrol.json).

## Gerçek uzman ekibi ve bağımsız notlar

Uzmanlar eşzamanlılık sınırına göre dalgalar halinde çalıştı; her biri ayrı agent görevi ve ayrı tamamlanma yanıtı üretti. Aşağıdaki 10 rolün tümü tamamlandı. [Görev/not/hash kaydı](uzman-is-kayitlari.json) bu eşlemeyi saklar.

| Rol | Gerçek agent görevi | Bulgu | Not |
| --- | --- | ---: | --- |
| 01 Mimari/durum | `/root/uzman_01_mimari` | 6 | [01-mimari.md](uzman-notlari/01-mimari.md) |
| 02 Veri/split | `/root/uzman_02_veri` | 5 | [02-veri.md](uzman-notlari/02-veri.md) |
| 03 Eğitim | `/root/uzman_03_egitim` | 5 | [03-egitim.md](uzman-notlari/03-egitim.md) |
| 04 Metrik/kalibrasyon | `/root/uzman_04_metrikler` | 5 | [04-metrikler.md](uzman-notlari/04-metrikler.md) |
| 05 Görüntü | `/root/uzman_05_goruntu` | 5 | [05-goruntu.md](uzman-notlari/05-goruntu.md) |
| 06 Deployment | `/root/uzman_06_deployment` | 6 | [06-deployment.md](uzman-notlari/06-deployment.md) |
| 07 Performans | `/root/uzman_07_performans` | 6 | [07-performans.md](uzman-notlari/07-performans.md) |
| 08 Dayanıklılık | `/root/uzman_08_dayaniklilik` | 5 | [08-dayaniklilik.md](uzman-notlari/08-dayaniklilik.md) |
| 09 Operatör | `/root/uzman_09_operator` | 5 | [09-operator.md](uzman-notlari/09-operator.md) |
| 10 Test/sürüm | `/root/uzman_10_test_surum` | 5 | [10-test-surum.md](uzman-notlari/10-test-surum.md) |

Uzman notlarında tam kaynak satırları, resmi sürümlü kaynak bağlantıları, probe komutları ve çıktıları bulunur. Aşağıdaki **Kaynak uzman** alanı her özgün bulgunun birleştirme eşlemesidir; 53 bulgunun hiçbiri sessizce düşürülmedi.

## Numara, önem ve kanıt sözleşmesi

P1: üretim kararını/kökenini veya artefakt korunmasını doğrudan etkileyen ilk işler; koşullu P1 kapasite/veri riskleri ayrıca etiketlidir. P2: kabul, işletim ve tekrar üretilebilirlik kusuru/riskleri. P3: bakım, dokümantasyon veya seçim politikası fırsatları. P0 bulgu saptanmadı. Öncelik, kanıt gücüyle aynı şey değildir.

**Doğrulanmış** ifadesi belirtilen kaynak yolu/sayısal fixture/mock sonucu içindir; native GUI'de ya da gerçek modelde ayrıca çalıştırıldığı anlamına gelmez. **Koşullu risk** için gerekli koşul yazılmıştır; gerçekleşmiş arıza/kalite düşüşü çıkarılmamıştır. **Fırsat** bir mevcut arıza iddiası değildir.

Satır kısaltmaları: **W** = [worker](../../worker/anomalib_worker.py), **UI** = [MainWindow](../../mainwindow.cpp), **QP** = [QProcess köprüsü](../../anomalibprocess.cpp), **CAM** = [kamera servisi](../../cameraservice.cpp). Satırlar inceleme sonundaki değişmemiş kaynaklara aittir.

### B01 — Üretim seçimi gerçek run ve dataset kökenine bağlanmıyor · P1

**Tür/kanıt:** Doğrulanmış kaynak kusuru; M01 + 09-01. UI:252/2353 başlangıçta demo kapalı olsa da sentetik referans sonuçları koyuyor; UI:3062 promotion tamamlanmış gerçek run/checkpoint/boş olmayan SHA kontrolü yapmıyor. Ayrıca A'nın sonuçları dururken dataset B seçilirse UI:3078 manifest kökünü güncel B'den alıyor. Eski iki demo production manifesti bulunması tarihsel destek; yeni GUI çalıştırması değildir.

**Tetikleyici/etki:** Temiz açılışta referans kazananı seçmek, checkpoint'i kaybetmek veya A eğitimi sonrasında B dataset'ine geçmek; sentetik/geçersiz model ya da yanlış veri kökeni üretim kaydına girebilir.

**Çözüm:** Immutable ExperimentResult içine real/demo, run ID, dataset/split kimliği, tamamlanma ve artifact fingerprint koymak. Promotion yalnız bu snapshot'tan üretilecek; adayın varlığı/hash'i/uyumu doğrulanmadan mevcut production manifest değişmeyecek.

**Kabul:** Demo kapalı temiz açılış üretim adayı üretmez; demo/eksik checkpoint/boş SHA reddedilir ve eski manifest korunur; A sonucu B seçiliyken kaydedilse bile köken A kalır veya açıkça reddedilir. **Efor:** 1–2 gün. **Kaynak uzman:** M01, 09-01.

### B02 — İkinci export aynı dosya ve manifestin üzerine yazabiliyor · P1

**Tür/kanıt:** Fake exporter ile gerçek path/manifest mantığında doğrulanmış. UI:3055 ortak export root; W:2579 filename vermiyor. Anomalib varsayılan `model` adıyla iki model aynı `weights/onnx/model.onnx` yoluna gidiyor; ikinci PaDiM export ilk PatchCore içeriğini değiştiriyor. Geçmiş kullanıcı modeli kaybı iddiası yok.

**Tetikleyici/etki:** Aynı root/formatta ikinci export; ilk kaydın kimliği artık aynı diskteki modele karşılık gelmez.

**Çözüm:** Run/model/fingerprint bazlı değişmez export dizini; staging içinde bütün dosyaları doğruladıktan sonra tamamlanmış manifest yayımlamak. **Kabul:** İki export sonrası ilk dosya/hash/manifest aynı kalır; ikinci export başarısızsa ilk kullanılabilir kalır. **Efor:** 0,5–1 gün; tam staging/completion 1–2 gün. **Kaynak uzman:** D06-01.

### B03 — Piksel karar haritası iki ayrı yerde kalibrasyonunu kaybediyor · P1

**Tür/kanıt:** İki bağımsız neden sayısal fixtures ile doğrulandı. (a) W:1801/1876 görüntü başına min–max; UI:1674/3932 kalibre float değer yerine quantized PNG'den threshold üretir. %60 eşiğinde düşük .1–.2 harita beklenen 0/4 yerine 2/4, yüksek .8–.9 beklenen 4/4 yerine 2/4, sabit .9 beklenen 4/4 yerine 0/4 piksel verir. (b) W:2967/3081 override=None olsa da graph'ın kalibre anomaly_map'ini saved-sigma ile blur edilmiş raw_map ile değiştirir; tiny CPU graph'ta .062745 → .250980, yaklaşık 4 kat. Yalnız min–max'ı kaldırmak (b)'yi çözmez.

**Tetikleyici/etki:** Dar/sabit değerli harita veya raw-map destekli ONNX; maske, overlay ve eşik anlamı bozulur. Görüntü düzeyindeki pred_score'un bu iki yolda doğrudan değiştiği iddia edilmiyor.

**Çözüm:** Karar için kalibre float haritayı saklamak; görsel kontrast ayarını ayrı yapmak; 8-bit PNG'yi tek karar kaynağı kullanmamak. Override kapalıysa graph çıktısını korumak; sigma override gerekiyorsa saklanan kalibrasyonu yeniden uygulamak veya sonucu yalnız tanılama olarak etiketlemek. Anomalib'in eşik/validation istatistiği normalizasyonu [resmi 2.6.0 PostProcessor](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/post_processing/post_processor.py) içinde tanımlıdır.

**Kabul:** Yukarıdaki üç fixture 0/4, 4/4, 4/4 verir; None/eşit sigma graph kalibrasyonunu korur; Torch/ONNX/OV aynı karar ölçeğinde karşılaştırılır. **Efor:** Ortak çözüm 2–3 gün ve ayrı gerçek model parity kabulü. **Kaynak uzman:** M04-01, D06-02.

### B04 — Eksik/geçersiz görüntü skoru karar üretimine girebiliyor · P1

**Tür/kanıt:** Sayısal fixture. W:1762 finite doğrulamıyor; W:1899 eksik score'u görüntü başına normalize map maksimumundan üretince düşük harita bile 1 olabiliyor. NaN Python karşılaştırmasında normal label dalına gidiyor; Qt'nin bozuk JSON'u kabul edip NORMAL gösterdiği doğrulanmadı.

**Tetikleyici/etki:** Backend'den None/NaN/Inf/boş/nonscalar skor; güvenilir olmayan anomaly/normal kararı veya bozuk olay.

**Çözüm:** Adapter sınırında sonlu scalar zorunlu; geçersiz tahminde açık hata ve kararsız durum. Fallback yalnız belgelenmiş kalibre ham skor sözleşmesiyle. **Kabul:** None/NaN/±Inf/nonscalar değerler karar ve artifact üretmez; geçerli düşük skor korunur. **Efor:** 0,5–1 gün. **Kaynak uzman:** M04-02.

### B05 — Model karşılaştırması tek veri ayrımına sabitlenmiyor · P1

**Tür/kanıt:** Metadata fixture + kurulu/resmi sürüm kaynağı. W:2260 her modele yeni datamodule oluşturuyor; W:2050 seed/split manifest vermiyor. Anomalib 2.6 Folder seed=None ve FROM_TEST ayrımı kullanır. 20 normal/8 anomaly dosyasıyla iki setup farklı dosyalar seçti; seed=42 tekrarlandı. Tek setup train/val/test kesişimi boştu; SAME_AS_TEST sızıntısı saptandığı söylenmemeli. [Resmi 2.6.0 Folder](https://github.com/open-edge-platform/anomalib/blob/lib/v2.6.0/src/anomalib/data/datamodules/image/folder.py).

**Tetikleyici/etki:** Çoklu model karşılaştırması/tekrar koşu; AUROC/F1 farkı yalnız model farkını temsil etmeyebilir. Veri setleri erişilemediği için gerçek winner sıralama değişimi ölçülmedi.

**Çözüm:** Karşılaştırma öncesi tek explicit train/validation/test manifesti üretmek ve bütün modellerde kullanmak; seed, dosya hash'leri ve grup kimliğini kaydetmek. Model/eşik seçiminden bağımsız son holdout ayırmak. **Kabul:** Modeller aynı split ID/dosyaları kullanır; aynı seed aynı sonucu verir; holdout seçime katılmaz; içerik değişimi manifest kimliğini değiştirir. **Efor:** 1–2 gün. **Kaynak uzman:** V02-01, M04-05.

### B06 — Aynı parça/çekim serisi farklı split'lere dağılabilir · koşullu P1

**Tür/kanıt:** Statik veri riski; UI:3926 timestamp dosya adı dışında parça/lot/session grubu saklamıyor. Dosya düzeyinde rastgele ayrım yakın kareleri/aynı parçayı ayırabilir; gerçek veri sızıntısı oranı ölçülmedi.

**Tetikleyici/etki:** Aynı parçanın tekrarlı kareleri, komşu seri veya kopyalar varsa bağımlı test örnekleri iyimser değerlendirme üretebilir.

**Çözüm:** Parça/lot/session/time metadata, exact/near-duplicate denetimi, group split ve ayrı zaman/lot holdout. **Kabul:** Hiçbir grup/hash train ile holdout arasında paylaşılmaz; raporda grup ve bağımsız örnek sayısı görünür. **Efor:** 1–3 gün + veri düzenleme. **Kaynak uzman:** V02-04.

### B07 — Managed varlık yenileme eski geçerli klasörü kaybedebiliyor · P2

**Tür/kanıt:** Yalnız yeni sentetik dosyalarda hata enjeksiyonu. W:1012 önce destination siliniyor, sonra staging replace; W:1043 promotion, W:1050 format/SHA doğrulaması sırası. Rename PermissionError'da eski klasör yok; validation hatasında bozuk aday managed konuma çoktan geçmiş.

**Tetikleyici/etki:** Eksik/bozuk lock'lı varlık yenilemede rename/format/hash/disk hatası; önceki managed cache kaybolur. Kullanıcı dataset'i veya özgün HF cache'inin silindiği iddia edilmiyor.

**Çözüm:** Tam doğrulamayı staging'de yapmak; versioned dizin + tamamlanmış manifest veya backup/rollback; yeni kopya ve lock başarılı olmadan eskiyi silmemek; asset root kilidi. **Kabul:** Validation/rename/lock-write hataları ve iki hazırlama isteğinde eski kullanılabilir varlık korunur; restart yarım staging'i ready saymaz. **Efor:** 1–2 gün. **Kaynak uzman:** 08-01.

### B08 — Geçersiz metrik JSON'u ve kazanan seçimini bozuyor · P2

**Tür/kanıt:** Sayısal probe. W:1707 NaN kabul ediyor; W:402 allow_nan açık JSON geçersiz token yayımlayabilir; QP:209 bunu olay yerine metin sayabilir. W:2325 yalnız None'ı dışlar; ilk NaN, sonraki .99'a rağmen winner olabilir; hepsi None iken ilk model seçilir.

**Tetikleyici/etki:** Tek sınıf/boş/undefined metric; sonuç paneli eksik kalır veya dayanaksız winner seçilir.

**Çözüm:** Tüm payload'ı finite/null'a normalize etmek, allow_nan=False; winner eligibility ve yeterli metric gereksinimi; geçerli karşılaştırma yoksa winner yok. **Kabul:** NaN/Inf içeren olay standard JSON; .99 geçerli aday seçilir; hepsi eksikse üretim adayı oluşmaz. **Efor:** 0,5–1 gün. **Kaynak uzman:** M04-03.

### B09 — Gösterilen F1 ve operatör eşiği farklı çalışma noktaları · P2

**Tür/kanıt:** Çalışma noktası riski; 2.6 postprocessor/pred_label ve UI:1871/W:1905. Modelin .5 etiketine göre F1 ile UI varsayılan .6 kararı farklı; .55 için anomaly etiketi UI'de normal olur. Bu eşiklerin ikisinin de her veri için yanlış olduğu iddia edilmiyor. [Resmi evaluator](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/components/base/anomalib_module.py).

**Tetikleyici/etki:** Eşik değiştirmek ve önceki F1'i geçerli kalite sonucu sanmak; yanlış alarm/kaçırma dengesi görünmez. Anomaly score olasılık değildir.

**Çözüm:** Kalibrasyon/eşik/operating-point kimliğini manifestte taşımak; seçilen eşiğe ait TP/FP/TN/FN, FPR/FNR'yi bağımsız holdout'ta raporlamak. **Kabul:** .55 fixture iki eşikte doğru farklı label verir; rapor hangi eşikte ölçüldüğünü gösterir; eşik değişince eski F1 yeni ölçüm diye sunulmaz. **Efor:** 1–2 gün + veri kabulü. **Kaynak uzman:** M04-04.

### B10 — Uint8 0/1 görüntü ONNX/OpenVINO'da yanlış ölçekleniyor · P2

**Tür/kanıt:** Girdi fixture'ı; W:2942/3055 ve legacy OV preprocessing `max>1` ise /255 uygular. Yalnız 0/1 değerli uint8 görüntü maksimumu 1 kalır, Torch'ta 1/255=.00392157 olur. Tek pikselin 2 olması bütün görüntünün dalını değiştirir; siyah 0 görüntü eşittir.

**Tetikleyici/etki:** Çok karanlık 8-bit görüntü; backend girdi ölçeği yaklaşık 255 kat ayrışabilir. Gerçek kalite etkisi ölçülmedi.

**Çözüm:** Dtype sözleşmesi: uint8 her zaman /255; float yalnız explicit [0,1] kontratıyla. **Kabul:** 0/1/2/255 fixtures bütün backend'lerde aynı tensor ölçeği verir; yanlış float aralığı reddedilir. **Efor:** 0,25–0,5 gün. **Kaynak uzman:** V05-01.

### B11 — Eğitim ve backend resize sözleşmesi eşdeğer değil · P2

**Tür/kanıt:** Doğrulanmış tensor farkı; kalite etkisi koşullu. Eğitim Resize antialias=True, Torch export_transform=False; ONNX/OV girişinde PIL bilinear resize W:2959/3075/3165. Seed42, 96×80→48×32 fixture'da train–Torch export normalized tensor MAE .39127, max 1.56889; train–PIL MAE .00474. Bu iki lossy resize yapıldığının veya üretim AUROC kaybının kanıtı değildir.

**Tetikleyici/etki:** Yüksek frekanslı/ince kusurlu görüntüler; backend değiştirmek farklı özellik/score üretebilir.

**Çözüm:** Versioned decode, renk, resize/antialias, dtype, normalization sözleşmesi; export uyumluluğunu kurulu sürümle açık seçmek. **Kabul:** Girdi tensor parity fixture'ları bağımsız oracle ile geçer; sonra aynı gerçek örneklerde score/map/label toleransı kaydedilir. **Efor:** 1–2 gün + gerçek model kabulü. **Kaynak uzman:** V05-02.

### B12 — Maske koordinatı ve preview kırpması tam görüntüyü açıklamıyor · P2

**Tür/kanıt:** UI kaynak geometrisi doğrulandı; kenar kusuru kaçırma etkisi koşullu. W:1965 maskeyi kaynak boyutuna döndürüyor; UI:1671/3953 raw model map'i kullanıyor. UI:1583 KeepAspectRatioByExpanding crop yapıyor. 640×480 kaynak ile 256×256 map farklı crop olur; 230×230 görünüm kaynak genişliğinin %25'ini gizleyebilir. Kırpma eski design-qa'da bilinçli tasarım kararıdır.

**Tetikleyici/etki:** Dikdörtgen/portre görüntü veya kenar kusuru; mask/input hizası ve operatör görünürlüğü sınırlanır.

**Çözüm:** Kaynak koordinatına registered float/binary harita; mask için nearest-neighbor; tam görüntü fit, gerekiyorsa belirgin crop ve full-size/zoom erişimi. **Kabul:** 4 köşe/merkez işareti 4:3, 16:9, portre ve DPI varyasyonunda hizalı; tam görüntüye erişilir. **Efor:** 0,5–1 gün; zoom 1–2 gün ek. **Kaynak uzman:** V05-03, V05-05.

### B13 — Kabul edilen 16-bit TIFF'in yoğunluk sözleşmesi yok · P2

**Tür/kanıt:** Fixture ile doğrulanmış dönüşüm, koşullu girdi riski. PIL RGB dönüşümü I16 [0,1,255,256,1024,32768,65535] değerlerini [0,1,255,255,255,255,255] yapıyor. TIFF kabul ediliyor; mevcut BGR8 kamera hattı bu riskin örneği değildir.

**Tetikleyici/etki:** 16-bit endüstriyel görüntü; yüksek değerler doygunlaşıp farklı yoğunluklar aynı görünür.

**Çözüm:** Desteklenmeyen bit depth'i açık reddetmek veya sensor/range metadata ile sabit dönüşüm; görüntü başına min–max kullanılmamalı. **Kabul:** Unsupported TIFF erken açık hata; destek varsa sabit kalibrasyonlu rampada beklenen değerler korunur. **Efor:** Ret 0,5 gün; destek 1–2 gün. **Kaynak uzman:** V05-04.

### B14 — Dataset yeterliliği eğitim başlamadan kontrol edilmiyor · P2

**Tür/kanıt:** Metadata setup fixture'ı; UI:2597 root/normal dir varlığını kontrol eder. Bir normal/sıfır anomaly Folder.setup'ta ValueError verdi.

**Tetikleyici/etki:** Çok küçük veya tek sınıflı veri; model başladıktan sonra anlaşılması güç hata, tanımsız iki sınıflı metric.

**Çözüm:** Manifest tabanlı class/split sayımı ve küçük decode/dimension preflight; normal-only destek varsa bunu ayrı sözleşme/metric politikasıyla sunmak. **Kabul:** Yetersiz ayrım model oluşturulmadan açıklanır; normal-only akış AUROC/F1 için iki sınıflı kalite iddiası yapmaz. **Efor:** 0,5–1 gün. **Kaynak uzman:** V02-02.

### B15 — Belirsiz basename fallback yanlış preview maskesi seçiyor · P2

**Tür/kanıt:** Sentetik metadata fixture; W:1530/UI:2564 global ilk stem eşleşmesine düşüyor. `abnormal/b/000` için birden fazla a/c maskesi varken a seçilebilir. Eğitim datamodule'u kendi eşlemesini kullanır; burada eğitim label'ının yanlış olduğu söylenmiyor.

**Tetikleyici/etki:** Alt klasörlerde aynı basename; operatör yanlış ground truth preview görür.

**Çözüm:** Tam göreli yol eşlemesi, ambiguity hata, boyut preflight. **Kabul:** Tek doğru göreli eşleşme seçilir; iki belirsiz adayda hiçbir maske otomatik alınmaz. **Efor:** 0,5–1 gün. **Kaynak uzman:** V02-03.

### B16 — Sayılan ve eğitime giren dosya kapsamları farklı · P2

**Tür/kanıt:** PNG+WEBP fixture; UI:2510/W:26 listesinde WEBP/PPM/PGM yok, W:2050 Folder'a explicit extensions vermiyor. Worker 1, Folder 2 dosya saydı.

**Tetikleyici/etki:** Ek desteklenen formatlar; UI sayısı ile split/eğitim sayısı uyuşmaz.

**Çözüm:** Tek manifest ve ortak extension/decode politikası, datamodule'a aynı kapsamı vermek. **Kabul:** UI/worker/Folder aynı dosya ID ve sayısını raporlar; dışlanan format görünür gerekçe taşır. **Efor:** 0,25–0,5 gün. **Kaynak uzman:** V02-05.

### B17 — EfficientAD teacher kanal ayarı hazır ağırlıklarla uyumsuz · P2

**Tür/kanıt:** Worker validation ve teacher ZIP opcode metadata ile doğrulanmış; UI:1111/1378, W:324 1–65536 kabul eder, hazır small/medium teacher çıkışı 384'tür. Worker 383'ü kabul ederken strict state load uyumsuz olur. Ağırlık tensoru yüklenmedi. [Resmi EfficientAD 2.6.0](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/efficient_ad/lightning_model.py).

**Tetikleyici/etki:** Hazır teacher ile 384 dışı değer; geç kabul edilmiş parametre eğitim kurulumunda hata verir.

**Çözüm:** Hazır teacher için 384 read-only; özel teacher varsa channel/normalization metadata sözleşmesiyle ayrı destek. **Kabul:** Small/medium 384 kabul, 383/385 erken ret; custom teacher yalnız uyumlu metadata ile. **Efor:** 0,5–1 gün. **Kaynak uzman:** 03-01.

### B18 — PaDiM feature sayısı seçilen layer kanal toplamını aşabiliyor · P2

**Tür/kanıt:** Validation + kurulu library AST dalı; W:310 geniş integer/None kabul ediyor. ResNet18 layer1 toplamı 64 iken otomatik 100 kütüphane ValueError verir. [Resmi PaDiM 2.6.0](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/padim/torch_model.py).

**Tetikleyici/etki:** Az layer seçimi veya büyük retained-features; UI kabul ettiği ayarla model kuramaz.

**Çözüm:** Backbone/layer kanal toplamından upper bound; otomatik min(default,total) politikası açık gösterilmeli veya erken ret. **Kabul:** Layer1'de 64 kabul, 65 ret; otomatik değer seçilen katmanlara uyumlu ve kayıtlı. **Efor:** 1 gün. **Kaynak uzman:** 03-02.

### B19 — Requested eğitim parametreleri effective diye kaydediliyor · P2

**Tür/kanıt:** Kurulu kaynak ve AST/stub probe; W:2262/2269 istenen max_epochs/devices'i verir, W:2318 bunları sonuca yazar. Anomalib PatchCore/PaDiM trainer_arguments 1 epoch/1 device uygulayabilir. İstenen 5/2 kaydının gerçek 1/1 çalışmayı temsil etmesi kusurdur; bu modellerin tek-pass davranışı tek başına hata değildir. [Resmi Engine 2.6.0](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/engine/engine.py), [PatchCore](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/patchcore/lightning_model.py).

**Tetikleyici/etki:** Override edilen trainer ayarı; deney karşılaştırması/yeniden üretim yanlış parametrelere dayanır.

**Çözüm:** Requested ve effective trainer/model parametre snapshot'larını ayırmak; override gerekçesi ve sürüm kimliğini kaydetmek. **Kabul:** 5/2 istekte sonuç her iki snapshot'ı içerir ve effective kurulu trainer'dan 1/1 okunur. **Efor:** 0,5–1 gün. **Kaynak uzman:** 03-03.

### B20 — EfficientAD scheduler step sayısı epoch aralığında uygulanıyor · P2

**Tür/kanıt:** Kurulu Anomalib 2.6.0 + Lightning 2.6.5 kaynak/AST ve iki scalar CPU scheduler kontrolü. StepLR step_size=`int(.95*max_epochs*batch_count)`; interval verilmediği için Lightning epoch varsayılanı uygular. 5 epoch/100 batch için step_size 475, yalnız 5 scheduler tick; beklenen geç eğitim decay oluşmaz. Model eğitilmedi; kalite etkisi ölçülmedi. [EfficientAD optimizer](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/efficient_ad/lightning_model.py), [Lightning 2.6.5 scheduler config](https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/2.6.5/src/lightning/pytorch/utilities/types.py).

**Tetikleyici/etki:** Bu sürüm çiftiyle EfficientAD eğitim; kaydedilen/plânlanan öğrenme hızı takvimi uygulanmaz.

**Çözüm:** Doğrulanmış pinned sürüm veya kontrollü adapter ile interval=step ve en az 1 step_size; site-packages elle değiştirilmemeli. **Kabul:** 500 optimizer-step planında decay 475. optimizer-step'te; küçük plan sıfır bölme/step_size hatası üretmez. **Efor:** 1–2 gün. **Kaynak uzman:** 03-04.

### B21 — Düşük PatchCore coreset oranı sıfır örneğe düşebiliyor · P2

**Tür/kanıt:** Koşullu küçük-veri riski; worker .001 kabul ediyor, library KCenter `int(embedding_count*ratio)` kullanıyor. AST örneğinde 128×.001 → 0, ret dalı. Tam eğitimde gerçekleşmiş hata ölçülmedi. [Resmi KCenterGreedy](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/components/sampling/k_center_greedy.py).

**Tetikleyici/etki:** Az embedding ve çok düşük ratio; masraflı feature toplamasından sonra coreset kurulamaz.

**Çözüm:** Erken tahmin ve gerçek bank sayısı üzerinden minimum 1 kontrol; ratio sessizce değiştirilmeyip açıklanmalı. **Kabul:** Sıfır üretilecek ratio mümkün olduğunca preflight'ta, kesinleşince coreset öncesi reddedilir; sınırdaki 1 örnek açık kayıtla çalışır. **Efor:** 0,5–1 gün. **Kaynak uzman:** 03-05.

### B22 — ONNX external-data dosyaları fingerprint dışında · P2

**Tür/kanıt:** Tiny CPU graph ile doğrulandı; W:633 yalnız .onnx hash'i, W:687 external-data load=False; büyük graph packaging W:2473 ve manifest W:2595 bütün ağırlık grubunu bağlamıyor. Dört bayt external weight 1→2 değişince output 1→2 oldu, fingerprint aynı; missing weight inspection'dan geçti. Mevcut 607 MB ONNX'in external-data kullandığı doğrulanmadı.

**Tetikleyici/etki:** External-data graph'ta weight değişmesi/kaybı; aynı fingerprint farklı model davranışını ya da çalışmayan teslimi temsil eder.

**Çözüm:** Graph ve her external file için path/size/hash manifesti; offset/length ve bütün dosya grubunu doğrulamak. **Kabul:** Ağırlık mutasyonu kimliği değiştirir veya yükleme reddedilir; eksik/bozuk external file model_loaded öncesi yakalanır. **Efor:** 1–2 gün. **Kaynak uzman:** D06-03.

### B23 — Model yükleme graph input/output/kimlik sözleşmesini tamamlamıyor · P2

**Tür/kanıt:** Üç farklı kabul açığı tiny CPU graph'larla doğrulandı. (a) W:688 input shape/type okumuyor, metadata yoksa W:725 256²; 1×3×384×512 graph doğru girdiyle çalışırken worker warmup'ta hata. FLOAT16 input'a W:2942 float32 gönderiliyor. Manifest `precision=float16` graph input dtype'ı demek değildir. (b) W:2939 warmup yalnız run; features-only output model_loaded alıp ilk predict'te map hatası veriyor. (c) W:528 schema/format semantiği ve W:693 model kimlik çelişkisi reddedilmiyor; hash doğru PaDiM manifest ile PatchCore heuristiği graph loaded olabiliyor. Heuristik tek başına model ailesinin matematiksel kanıtı değildir.

**Tetikleyici/etki:** Metadata'sız/özel graph, yanlış sidecar veya eksik output; geç yükleme/tahmin hatası ya da yanlış family/capability kaydı.

**Çözüm:** Input layout/dtype/fixed-dynamic boyut/required inputs, output finite score + 2D map/shape ve versioned manifest consistency load kabulünde doğrulanmalı. Güvenilir kimlik belirsizse bunu açık taşımak; shape ve sidecar çelişkisini erken reddetmek.

**Kabul:** 384×512/FLOAT16 sözleşmesi doğru tensor ile desteklenir veya açık unsupported hatası; features-only warmup'ta reddedilir, eski model korunur; bilinmeyen schema ve güvenilir metadata çelişkisi loaded olamaz. **Efor:** 2–3 gün ortak adapter/kabul. **Kaynak uzman:** D06-04, D06-05, D06-06.

### B24 — İlk promotion düğmesi sonraki kazanan için açılmıyor · P2

**Tür/kanıt:** Statik UI kusuru; UI:3105 düğmeyi kapatıyor, UI:2646/3001 yeni eğitim/winner için yeniden açmıyor.

**Tetikleyici/etki:** A'yı promote edip aynı oturumda B eğitimini tamamlamak; geçerli ikinci aday UI'den seçilemez.

**Çözüm:** Düğme durumunu current valid artifact ID ile production ID farkından türetmek. **Kabul:** A promote → B valid result sonrası düğme açık; zaten production olan aynı aday için gerekçeli kapalı. **Efor:** 0,5–1 gün. **Kaynak uzman:** M02.

### B25 — Workspace ve kalıcı depo kimliği EXE/Python yoluna bağlı · P2

**Tür/kanıt:** İki statik koşullu risk: UI:3966 dataDirectory EXE/.data, UI:4011 model assets workspace/.data; UI:3993 workspace seçimi Python EXE atasını önceleyebiliyor. A uygulamasında B venv seçmek assets/relative ImageNette kökünü B'ye, sonuçları A'ya bağlayabilir. mkpath sonucu denetlenmiyor; salt okunur install klasöründe hata riski. Mevcut veri kaybı doğrulanmadı.

**Tetikleyici/etki:** Debug/release veya farklı klasörden çalıştırma, Python değiştirme, installed EXE; kayıtlar bölünür veya yazılamaz, QSettings son model kaydı başka depoyu işaret edebilir.

**Çözüm:** Python'dan bağımsız explicit workspace ID; kullanıcı-yazılabilir kalıcı depo/QStandardPaths; görünen resolved paths ve kontrollü, eski depoları koruyan migration. **Kabul:** Aynı workspace iki build'de aynı veriyi görür; farklı Python workspace kimliğini değiştirmez; readonly EXE klasöründe yazılabilir depo seçilir; migration önceki veriyi korur. **Efor:** 1–2 gün + migration kabulü. **Kaynak uzman:** M03, M04.

### B26 — Demo/gerçek geçişi aktif inference state'ini taşımaya devam ediyor · P2

**Tür/kanıt:** Statik durum kusuru; UI:2027 toggle settings'i değiştiriyor; UI:3256 demo winner backend olmadan ready=true. UI:2409 ready ise real load'u atlayabiliyor, W:3241 backend yok hatası; UI:3448 predict hata dalı ready'yi tekrar true yapabilir. Ters geçişte gerçek model etiketi synthetic sonucu açıklayabilir. Native akış çalıştırılmadı.

**Tetikleyici/etki:** Demo winner sonrası demo off veya yüklü real model sonrası demo on; görünen hazır durum/mod/model ile çalışacak backend ayrışır.

**Çözüm:** Mode/session generation ve backend kimliğini readiness'e bağlamak; toggle'da pending/result state invalidation; gerçek loaded acknowledgement olmadan real ready yok. **Kabul:** Demo→real backend yüklenir; real→demo synthetic sonuç açık demo kimliği taşır; backend yokken predict hata sonrası ready oluşmaz. **Efor:** 1–2 gün. **Kaynak uzman:** 09-03.

### B27 — Model yükleme olayları aktif request'e bağlanmıyor · P2

**Tür/kanıt:** Statik, koşullu UI olay sırası riski; UI:3207 tek pending path/had-ready bayrağı; UI:3361 request_id eşleştirmeden model_loaded pending'i temizler. Load/predict sendCommand dönüşleri UI:3234/3592 kontrol edilmez. Worker seri çalışır; paralel backend yürütme iddiası yok.

**Tetikleyici/etki:** A hazır → B load → C load kuyruğu → B success → C error; B backend'i korunurken UI ready=false olabilir, ya da eski olay C pending'ini kapatır. IPC write failure bekleyen düğmeler bırakabilir.

**Çözüm:** Aktif load/predict request ID + session generation; yalnız ilgili terminal event; bir load sırasında ikinci load'u kapatmak veya explicit queue; worker aktif fingerprint/loaded availability bildirsin. **Kabul:** B/C senaryosunda B kullanılabilir kalır; eski session/request yeni pending'i temizlemez; write failure bekleme bırakmaz. **Efor:** 1–2 gün. **Kaynak uzman:** 08-02.

### B28 — Eski sonuç paneli yeni girdi/model için geçerli gibi kalabiliyor · P2

**Tür/kanıt:** Statik belirli erken dönüşler; UI:1918 editable input'ta textChanged invalidation yok; UI:3516–3530 erken hata öncesi sonuç temizlenmiyor. Model loaded UI:3361 eski score/map'i koruyor. Browse ve normal predict başlangıcı ise zaten temizliyor; tüm hata yolları için aynı kusur iddia edilmiyor.

**Tetikleyici/etki:** M1/A normal sonrası manuel okunamayan B yazmak ve test etmek; veya M2 yükleyip M1 sonucunu panelde görmek. Footer hata/model değişse de eski sonucu current sanmak mümkün.

**Çözüm:** Son tahmini input/artifact/request/mode/threshold/sigma snapshot'ıyla tutmak; seçim/değer değişiminde temizlemek veya görünür historical işareti; erken ret current decision'ı geçersiz kılsın. **Kabul:** Manuel geçersiz input ve model değişiminde eski NORMAL current gösterilmez; sonuç provenance'ı görünür. **Efor:** 1–2 gün. **Kaynak uzman:** 09-04.

### B29 — Rescan ve Capture farklı dataset hedefini kullanabiliyor · P2

**Tür/kanıt:** Statik UI veri akışı; Browse UI:2425 iki alanı eşitler; Rescan UI:2440 m_datasetRoot'u güncellerken Capture edit/label'ı eşitlemez. UI:3911 Capture eski A edit'ini alıp merkez root'u A'ya döndürebilir. Kamera ile gerçek yazma çalıştırılmadı.

**Tetikleyici/etki:** Browse A → manuel B/Rescan → Capture; yeni görüntü beklenen B yerine A'ya gidebilir.

**Çözüm:** Tek dataset setter veya bilinçli ayrı/daima görünür capture target; capture merkezi dataset seçimini yan etkiyle değiştirmesin. **Kabul:** Camera stub ile B seçimi sonrası hedef B; ayrı hedef A amaçlanıyorsa A/B açık görünür ve merkezi B korunur. **Efor:** 0,5–1 gün. **Kaynak uzman:** 09-05.

### B30 — Ortam doğrulama demo/keşif/yürütme kapsamını ayırmıyor · P2

**Tür/kanıt:** Demo probe + statik gerçek run_check. Demo varsayılan true UI:2021; README önce Verify diyor. UI:3679 demo check'e gider, W:1603 Anomalib/Torch import etmeden ok=true döner; mesaj Demo worker ready der, tamamen gizlenmiş olduğu iddia edilmez. Gerçek W:3379/3396 provider listesinde CUDA varsa CUDA provider ready der; Session/run yapılmaz. GPU'nun bozuk olduğu kanıtlanmadı. [ORT resmi Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).

**Tetikleyici/etki:** Kurulum kabulünde Verify'ı gerçek GPU execution sanmak; DLL/driver/kernel sorunu ilk load/run'a kadar görünmeyebilir.

**Çözüm:** Verify gerçek ortamı denetlesin; demo testi ayrı ve açık scope'lu olsun. Detected, runtime_smoke_passed, model_loaded ve model parity kabulünü ayırmak; birkaç KB graph ile ayrı runtime smoke, fallback GPU başarısı sayılmadan.

**Kabul:** Demo açıkken Verify gerçek import/ortam kapsamını bildirir; CUDA listelenip Session/run başarısızsa GPU smoke başarısı yok; fallback ayrı kayıt; README sırası akışla tutarlı. **Efor:** Mesaj/akış 0,5–1 gün; ayrı CPU/CUDA smoke 1 gün. **Kaynak uzman:** 09-02, T10-04.

### B31 — Backendler aynı latency alanında farklı işleri ölçüyor · P2

**Tür/kanıt:** Timer kontratı fake clock ile doğrulandı. Checkpoint W:2779 Engine.predict decode/dataloader/trainer dahil; ONNX W:2957/OV W:3073/3163 timer decode/resize sonrası. Aynı model_latency_ms adı altında 15 ms fake decode + 5 ms fake forward, 20 vs 5 ms verir. Bunlar gerçek gecikme ölçümü değildir. W:3260 worker total'i Qt/kamera/IPC/render kapsamıyor; mevcut CUDA synchronize olumlu ve korunmalı.

**Tetikleyici/etki:** Backend hızını tablo üzerinden kıyaslamak; karşılaştırma kapsam farkına dayanır, operatör toplam süresi görünmez.

**Çözüm:** Decode/preprocess/forward/postprocess/write ve worker_total/operator_total phase sözleşmesi; soğuk başlangıç ayrı; senkron GPU ölçümü. **Kabul:** Aynı fake clock bütçesi bütün backendlerde aynı phase değerlerini verir; toplam sınırı/donanım/örnek sayısı raporlanır. **Efor:** Alan/etiket 0,5 gün; ortak ölçüm 1–2 gün. **Kaynak uzman:** 07-01.

### B32 — ONNX CPU fallback sonrasında GPU etiketi eski kalıyor · P2

**Tür/kanıt:** Fake session ile doğrulanmış diagnostic kusur; W:2917 constructor ilk provider'dan devicecuda kuruyor; warmup/run EPFail ile CPU'ya düşerse W:3277 etiketi güncellenmiyor. Gerçek GPU fallback bu makinede oluşturulmadı.

**Tetikleyici/etki:** ORT runtime provider reset/fallback; CPU çalışmasına GPU latency/cihaz beyanı eşlik eder. Provider listesinin ilk CUDA olması bütün node'ların GPU çalıştığını kanıtlamaz.

**Çözüm:** Preferred ve active provider'ı ayırmak; run sonrasında actual session/provider ve fallback reason; gerekirse ORT profiling node yerleşimi. **Kabul:** CUDA→CPU mock geçişinde event/UI CPU ve fallback bildirir; sessiz GPU başarısı yok. **Efor:** 0,5–1 gün. **Kaynak uzman:** 07-02.

### B33 — Model değişiminde iki backend aynı anda bellekte · koşullu P1

**Tür/kanıt:** W:3193 yeni candidate kurulurken eski backend W:3219'a kadar tutulur; weakref fixture eşzamanlı ömrü doğruladı. Öncekini hata durumunda korumak olumlu davranış. Diskteki 607 MB dosya VRAM boyutu değildir; gerçek OOM ölçülmedi.

**Tetikleyici/etki:** Eski+aday model ve runtime workspace bütçesi aşılırsa model değişimi OOM yapabilir; operatör geçişi tamamlayamaz.

**Çözüm:** Bellek bütçesi ve ölçüm; normal atomic load ile düşük bellek unload-first/reload-rollback politikalarını açık seçmek; başarısızlıkta kullanılabilir backend durumunu bildirmek. **Kabul:** Kısıtlı fake memory budget aday yüklemeden ret/politika seçimi sağlar; atomic failure eski modeli korur; unload-first failure rollback sonucu açık. Gerçek peak RSS/VRAM sonraki izinli kabulde ölçülür. **Efor:** 1–2 gün. **Kaynak uzman:** 07-03.

### B34 — PatchCore feature bank için kapasite planı yok · P2

**Tür/kanıt:** Koşullu kapasite riski; UI 2048² ve ratio=1 kabul edebilir. PatchCore coreset öncesi tüm embedding'leri toplar; ratio düşürmek ilk feature toplama belleğini düşürmez. Alt sınır tahmini N×P×D×b; bank ratio×N×P×D×b, chunk distance ek bellek. 256→2048 iki eksende patch alanı yaklaşık 64 kat; bu 64 kat latency ölçümü değildir.

**Tetikleyici/etki:** Büyük çözünürlük/çok görüntü/büyük ratio; geç OOM veya beklenmedik uzun çalışma. Gerçek kapasite sınırı test edilmedi.

**Çözüm:** Dataset/model layer/precision ile preflight estimate ve overhead payı; eğitimde peak RSS/VRAM kaydı; gerekiyorsa desteklenen yöntemle streaming/feature azaltımı. **Kabul:** Aynı N/P/D/b fixture'ı tutarlı estimate verir; bütçe aşımı model başlamadan açıklanır; gerçek ölçüm estimate'den ayrı görünür. **Efor:** Plan 1–2 gün; ölçüm 0,5 gün. **Kaynak uzman:** 07-04.

### B35 — Gerçek sonuçlar karşılaştırılabilir inference benchmark taşımıyor · P2

**Tür/kanıt:** Fırsat; W:2309 gerçek train latency=None; UI karşılaştırma sütunu var. Küçük JSON okumasında 35 gerçek export-job kaydı latency null; süreleri fit+test süreleri, inference/export latency'si değildir. Eski 38,2 ms production değeri demo kökenli; dosya kopyaları bağımsız eğitim sayılmaz.

**Tetikleyici/etki:** En iyi kalite/hız modelini seçmek; gerçek ortak latency kanıtı yoktur.

**Çözüm:** Sabit örnek manifesti, warmup/soğuk load ayrımı, n/p50/p95, phase/device/backend/fingerprint ve güç profiliyle benchmark; B31 düzeltildikten sonra. **Kabul:** Her aday aynı örneklerle ölçülür; eksik sonuç N/A; demo süreleri gerçek performans diye karşılaştırılmaz. **Efor:** 1–2 gün + gerçek model ölçümü. **Kaynak uzman:** 07-05.

### B36 — OpenVINO cihaz seçimi artefakt yoluna göre değişiyor · P3

**Tür/kanıt:** Fırsat; W:2998 ONNX yolu ilk derlenebilen GPU→CPU, W:3115 IR yolu üç predict/two-higher benchmark ile seçer; cache fingerprint tabanlı. Gerçek en hızlı cihazın yanlış seçildiği ölçülmedi.

**Tetikleyici/etki:** Aynı modelin ONNX/IR yolu, runtime/donanım/güç profili değişimi; tekrar üretilebilir seçim politikası belirsiz.

**Çözüm:** Explicit CPU/GPU/AUTO veya tutarlı benchmark politikası; cache kimliğine runtime/hardware/power/policy eklemek. **Kabul:** Aynı policy iki formatta aynı seçim gerekçesi; donanım/runtime değişimi eski cache'i geçersiz kılar. **Efor:** 0,5–1,5 gün. **Kaynak uzman:** 07-06.

### B37 — Süreç terminal durumu ve UI beklemeleri birlikte düzenlenmeli · P2

**Tür/kanıt:** İki statik koşullu risk. QP:143–166 stop wait/terminate/kill sonrası son kill bitişi onaylanmadan UI:2664 Stopped olur; processError türleri tek jobFailed'e dönüşür; start timeout süreç state'ini ayrıca kapatmaz. UI kamera slotları CAM:86 RetrieveResult(5000) çağrısını senkron yapar; QP start/stop/close wait'leri ana thread'dedir. QProcess destructor beklediği için kalıcı orphan çıkarımı yok. [Qt 6.11 QProcess](https://doc.qt.io/qt-6.11/qprocess.html), [Basler CInstantCamera](https://docs.baslerweb.com/pylonapi/cpp/class_pylon_1_1_c_instant_camera).

**Tetikleyici/etki:** Yavaş startup, stdin/write error, uzun load/command veya kare/trigger yokluğu; UI erken terminal duruma geçebilir veya olay döngüsü bekler. 5 s capture ve 0,5+1,5+2 s shutdown bütçeleri normal latency ölçümü değildir. Zorunlu stop Python finally temizliğini garanti etmez.

**Çözüm:** Starting/Running/Stopping/Stopped/Failed state machine; terminal yalnız finished/NotRunning ile; cancellation ayrı; started/finished/stateChanged + timer tabanlı asenkron process akışı. Kamera açık sahiplik/iptal ve kapanışta görev bitişiyle worker thread'e taşınmalı; yarım artefakt başarı sayılmamalı.

**Kabul:** Yavaş-start/stdin-close/yanıtsız-shutdown küçük child süreçlerinde düğmeler yalnız gerçek terminalde açılır; 5 s fake kamera beklerken UI heartbeat ilerler; kapanış görev ömrünü kapsar. Gerçek kamera kabulü bu incelemede yapılmadı. **Efor:** Ortak state/process/kamera işi 2–4 gün. **Kaynak uzman:** 08-03, 08-05.

### B38 — Eksik output_root `.` olarak kabul ediliyor · P2

**Tür/kanıt:** Protokol probe; W:3257 boş alan önce Path('') olur, string '.' boş kontrolünü geçer. Mock writer'a '.' gönderildi; gerçek çalışma klasörüne yazılmadı. Normal Qt komutu UI:3581 alanı verir.

**Tetikleyici/etki:** Elle/gelecekteki istemciden eksik/boş path; W:1877 sabit PNG adları çalışma klasörüne yazılabilir/üzerine yazılabilir.

**Çözüm:** Path oluşturulmadan string/nonempty/whitespace/absolute politika; benzersiz request output dizini ve tamamlanmış artifact manifesti. **Kabul:** Eksik/boş/whitespace predict backend'e gitmeden ret ve sıfır artifact; geçerli absolute review path kabul; göreli yol politikası explicit. **Efor:** 0,25–0,5 gün; completion manifest 0,5 gün ek. **Kaynak uzman:** 08-04.

### B39 — Tekrar kurulabilir tam ortam manifesti yok · P2

**Tür/kanıt:** Koşullu tekrar üretilebilirlik riski; README:63 sürüm seçimi serbest, requirements-inference yalnız ORT bandı. W:146 geniş Anomalib major/minor bandı; test 2.9.0rc1 kabul eder. `.pro` native SDK/DLL yolları makineye bağlı. Mevcut pip check temiz; mevcut bandın kesin bozuk olduğu ileri sürülmüyor.

**Tetikleyici/etki:** Yeni makine, venv kaybı veya kitaplık güncellemesi; aynı job farklı API/preprocessing/scheduler/runtime davranışı verebilir.

**Çözüm:** Desteklenen CPU/CUDA platform profilleri; üst/transitif sürüm ve wheel/hash lock, CUDA wheel kaynağı, native Qt/MSVC/Pylon/OpenCV/DLL listesi; interpreter/runtime manifest hash'i run/export kaydında. [pip tekrarlanabilir kurulum](https://pip.pypa.io/en/stable/topics/repeatable-installs/).

**Kabul:** Temiz desteklenen makine yalnız manifestten kurulur; pip check/suite/tiny CPU graph kabulü geçer; CPU/CUDA profilleri ayrı sonuç taşır. Bu incelemede temiz kurulum yapılmadı. **Efor:** Manifest 1 gün; temiz ortam matrisi 1–2 gün ek. **Kaynak uzman:** T10-01.

### B40 — Test edilen kaynak worker paket kimliğine bağlanmıyor · P2

**Tür/kanıt:** Koşullu paket riski; test kaynak worker'ı, UI:4024 EXE resource'undan çıkarılan worker'ı kullanır. Resource farklıysa cache atomik yenilenir; bu olumlu koruma eski EXE'yi yeni kaynağa bağlamaz. Generated 4 resource'un 3'ü güncel 130118 B / SHA başlangıcı bf9fe0b1a3929909 kaynaktan farklı: agent-debug 126272 B, agent-release 120088 B, Desktop Debug 17372 B; Desktop Release aynı. EXE linked/embedded içerikleri doğrudan doğrulanmadı. [Qt resource build davranışı](https://doc.qt.io/qt-6.11/resources.html).

**Tetikleyici/etki:** Kaynak değişiminden sonra eski EXE/farklı build kullanmak; suite'in kapsadığı worker ile operatör worker'ı farklı olabilir, uygulama 0.1.0 etiketi ayırt etmez.

**Çözüm:** Native build ID, kaynak manifest/worker SHA ve JSON protocol version; handshake/result/export'ta kimlik; paket kabulünde resource ile test hash karşılaştırması, tek dağıtım çıktısı. Tarihsel build'leri silmek çözüm kabul edilmemeli.

**Kabul:** Source değişmiş/resource eski aday release kabulünde ret; güncel paket çıkarılan worker hash'i test kaydına eşit; bilinmeyen protokol açık hata. **Efor:** Kimlik + paket kontrolü 1–2 gün. **Kaynak uzman:** T10-02.

### B41 — 37 test kritik karar/export sözleşmelerini henüz kapsamıyor · P2

**Tür/kanıt:** Test fırsatı; suite gerçek constructor/CPU blur parity dahil yararlı kontroller taşır, bütün testler mock değildir. Ancak B02/B03/B04/B08/B22/B23 somut kusurları mevcut suite geçerken yeniden üretildi; gerçek kalibrasyon/output/metadata'sız shape/dtype/external weight kabulü boşlukta.

**Tetikleyici/etki:** Suite'i tek release kapısı saymak; aynı kusur regresyon olarak geri gelebilir. Testlerin geçmesi başarı/kalite oranı değildir.

**Çözüm:** Düzeltmelerle birlikte bu bağımsız probes'ları outcome regresyonlarına dönüştürmek; beklenti uygulama fonksiyonundan hesaplanmamalı. Hızlı CPU/sayısal kabul ile native UI/gerçek model/hardware kabulünü ayrı koşular tutmak.

**Kabul:** Her yeni test mevcut kusurlu kaynak üzerinde ilgili nedenle fail, çözülmüş kaynakta pass; ilk export korunması, kalibre maske, finite score/JSON, external weight, fixed shape/type ve load output kontratı bağımsız oracle ile; mevcut 37 test korunur. **Efor:** İlk 5–8 test 1–2 gün. **Kaynak uzman:** T10-03.

### B42 — README güncel test/kabul kapsamını yanlış tarif ediyor · P3

**Tür/kanıt:** Dokümantasyon kusuru; README:217 26 test derken kaynak/kayıt 37. Persistent process/one-shot/future continuous deployment anlatımı belirsiz; sürekli kamera loop'u mevcut diye yorumlanmamalı. Geçmiş native/kamera passed beyanları build/hash/tarih kaydına bağlı değil; geçmiş testlerin yanlış olduğu ileri sürülmüyor. Demo doğrulama sırası B30'da çözülecek.

**Tetikleyici/etki:** README'yi güncel release kabul kaydı sanmak; test sayısı/sürüm/donanım kapsamı karışır.

**Çözüm:** Test sayısını suite'ten üretmek veya yalnız komut/kapsam; release build/worker hash, tarih, ortam, gerçek command/result ve donanım erişimi; persistent backend, tek görüntü isteği ve sürekli kamera ayrı anlatılmalı.

**Kabul:** Doküman 37 teste ve mevcut `--serve` mimarisine tutarlı; tarihsel kamera/native kayıtları yeni kontrol diye gösterilmez. **Efor:** 0,5 gün. **Kaynak uzman:** T10-05.

### B43 — Domain state ve parametre sözleşmesi UI/iki dilde çoğalıyor · P3

**Tür/kanıt:** İki bakım fırsatı; 4190 satırlık MainWindow UI/domain/path/artifact/controller'ı birlikte yönetiyor. UI:93/1215 ve W:26/199/260 parametre listesi/default/validator'ı ayrı tutuyor. Bugün bütün default'larda drift doğrulanmadı; eski sonuçta eksik parametre UI:2967 güncel default ile doldurulabilir.

**Tetikleyici/etki:** Yeni model/parametre ve UI değişimi; durum/validator drift riski, izole test güçlüğü. Sayısal bakım maliyeti ölçülmedi.

**Çözüm:** Önce WorkspacePaths, typed ExperimentResult/ModelArtifact ve kritik promotion/inference controller'larını ayırmak; versioned ortak JSON parametre şeması veya contract fixtures; eksik eski değeri unknown/legacy olarak taşımak.

**Kabul:** Promotion/mode/request/path geçişleri GUI olmadan test edilir; C++/Python default/range/serialization aynı fixtures ile eşleşir; legacy eksik değer etkin değer gibi yazılmaz. **Efor:** Kritik ayrıştırma 2–4 gün; contract 1–2 gün / tam schema 3–5 gün; bütün UI ayrıştırması 5–10 gün ayrı kapsam. **Kaynak uzman:** M05, M06.

## Önerilen uygulama ve kabul sırası

1. **Karar ve dosya koruma:** B01–B04, B07–B08, B22; bunların outcome testleri B41 ile aynı değişikliklerde. Release kapısı: sahte köken, overwrite ve kalibrasyon/finite kusuru tekrar üretilemiyor; önceki geçerli artefaktlar hata durumunda korunuyor.
2. **Değerlendirme güveni:** B05–B06, B09, B14–B16. Tek manifest ve holdout kimliği oluşmadan yeni model karşılaştırmasının kazanım iddiası yapılmamalı. Gerçek veride group/hash ve mask pairing preflight ayrıca yapılacak.
3. **Model/girdi sözleşmesi:** B10–B13, B17–B21, B23. Küçük CPU fixtures geçince, aynı kayıtlı gerçek örnekler/checkpoint/exports ile score/map/label parity ve seçilmiş eşikte FPR/FNR ölçümü planlanacak. Bu gerçek model işi bu analizde yürütülmedi.
4. **Operatör ve süreç:** B24–B30, B37–B38; typed snapshot ve request generation ortak değişiklik olarak. Fake native child/camera state kabulü, sonra ayrı donanım kabulü. Dataset/capture/production kimlikleri birbirinden tutarsız değişmemeli.
5. **Kapasite ve ölçüm:** B31–B36. B33/B34 büyük model yükleme/eğitim öncesi zorunlu plan; gerçek benchmark B31 ölçüm sözleşmesi ve B30 runtime scope ayrımı sonrası. Model/hardware fingerprint, warmup, örnek sayısı, p50/p95 ve peak RSS/VRAM ile kayıt.
6. **Dağıtım ve bakım:** B39–B43. Kilitli destek matrisi, test edilmiş worker resource hash'i, native paket kimliği ve doğrulanmış release kaydı; aşamalı refactor. Dokümantasyon düzeltmeleri ilgili davranış değişiklikleriyle birlikte yapılmalı.

Bu sıra tüm kodu yeniden yazmayı gerektirmez. Önce küçük ve bağımsız acceptance testleriyle karar/artefakt sözleşmeleri sabitlenmeli; controller/schema ayrımı bu sözleşmeleri koruyacak şekilde ilerlemeli.

## Açık kalan doğrulamalar

Gerçek veri olmadığından model seçimi, AUROC/F1/FPR/FNR, near-duplicate/grup sızıntısının gerçekleşip gerçekleşmediği ve eşik kalibrasyonunun saha etkisi açık. Büyük ONNX'in gerçek graph shape/input dtype/output kimliği, external-data durumu, GPU/OV parity'si ve peak bellek/latency'si açık. Native EXE'nin tam embedded worker SHA'sı, güncel derleme, Qt olay sırası, gerçek kamera formatı/trigger/donanım gecikmesi ve shutdown davranışı açık. Bunlar rapordaki statik veya tiny CPU testlerden türetilerek tamamlanmış sayılmadı.

Bir sonraki uygulama çalışması, veri/model/donanım kabul kapsamını ayrıca tanımlamalı. Buradaki dosyalar yalnız inceleme çıktısıdır; uygulama kaynakları ve mevcut veri/model/sonuçlar korunmuştur.

## Ek probe ve koruma kanıtları

| Alan | Birincil çıktı |
| --- | --- |
| Veri/split | [metadata probe](kanitlar/veri-metadata-kontrol.json), [iş özeti](kanitlar/veri-is-kayit-ozeti.json) |
| Eğitim | [AST/scheduler/teacher metadata logu](kanitlar/03-egitim-probe.log) |
| Metrikler | [sayısal probe](kanitlar/probe-04-metrics-result.json) |
| Görüntü | [girdi/dtype/resize probe](kanitlar/05-goruntu-probe.json) |
| Deployment | [tiny CPU graph/fake export probe](kanitlar/probe-06-deployment-result.json) |
| Performans | [fake clock/provider/lifetime probe](kanitlar/07-performans-probe.json) |
| Dayanıklılık | [hata enjeksiyonu/protokol probe](kanitlar/08-dayaniklilik-probe.json) |
| Operatör | [demo environment probe](kanitlar/operator-demo-check-probe.json) |
| Paket kimliği | [resource/hash probe](kanitlar/10-surum-probe.json) |
| Başlangıç koruması | [25 kaynak SHA baseline](kanitlar/kaynak-baseline.json), [2.550 dosya metadata baseline](kanitlar/veri-build-envanter.json) |
| Son koruma kontrolü | [sonuç](kanitlar/baseline-son-kontrol.json), [yeniden çalıştırılabilir kontrol](kanitlar/baseline-son-kontrol.ps1) |

Koruma sonucu: 25 küçük kaynak/belge/test/asset/QA dosyasının SHA-256, boyut ve UTC mtime'ı başlangıca eşit. `.data/build` altındaki 2.550 dosya yol/boyut/UTC mtime bakımından eşit ve bu iki ağaca yeni dosya eklenmedi. Büyük ikililer için içerik hash taraması yapılmadı; kontrol tüm `.venv` veya baseline dışı dosyalar hakkında bir içerik bütünlüğü iddiası taşımaz. Raporlar ve yeni sentetik kanıtlar yalnız bu inceleme dizininde oluşturuldu.
