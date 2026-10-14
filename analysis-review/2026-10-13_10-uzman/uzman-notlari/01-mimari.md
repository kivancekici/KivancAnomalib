# Uzman 01 — Mimari ve bakım kolaylığı

İnceleme tarihi: 13 Ekim 2026. Rol bağımsız kaynak incelemesiyle tamamlandı. Kaynak/config dosyaları değiştirilmedi; uygulama, kamera, eğitim, export veya indirme çalıştırılmadı.

## Kapsam ve mevcut durum

`mainwindow.cpp/.h`, `anomalibprocess.cpp/.h`, `cameraservice.cpp/.h`, `main.cpp`, `resources.qrc`, `KivancAnomaLibTest.pro`, `README.md` incelendi. UI/worker sözleşmesi için `worker/anomalib_worker.py` içindeki varsayılanlar, parametre doğrulama, manifest çözümü ve workspace yolu okundu. `tests/test_worker.py` test adları/kapsamı tarandı; test suite bu rol tarafından çalıştırılmadı.

Olumlu temel: Python işleri Qt sürecinden `QProcess` + JSON Lines üzerinden ayrılıyor; sürekli inference ayrı süreç kullanıyor; iş dosyası, worker güncellemesi ve üretim manifesti `QSaveFile` ile yazılıyor. Kaynak UI tarafında 4.190 satırlık `mainwindow.cpp`, Python tarafında 3.487 satırlık tek worker bulunuyor. Bu sayılar `Get-Content ... .Count` ile ölçüldü; performans ölçümü değildir.

## M01 — Sentetik veya geçersiz sonuç üretim modeli olarak işaretlenebiliyor

- **Tür / önem:** Doğrulanmış hata, **P1**. Doğrulama kaynak akışı ve mevcut disk artefaktıyla yapıldı; yeni uygulama oturumu çalıştırılmadı.
- **Kanıt:** `mainwindow.cpp:252` açılışta koşulsuz `populateReferenceState()` çağırıyor. `mainwindow.cpp:2353–2392` sentetik veri sayıları, AUROC/F1/gecikme ve `demo/.../model.ckpt` yollarını gerçek sonuçlarla aynı `m_results` alanına yerleştiriyor. Demo ayarı `mainwindow.cpp:2021–2030` içinde yalnız kaydediliyor; açılış referanslarını veya sonuç provenance'ını değiştirmiyor. `mainwindow.cpp:721–725` üretime yükseltme düğmesini oluşturup bağlarken devre dışı bırakmıyor. `mainwindow.cpp:3062–3106` demo bayrağı, tamamlanmış gerçek eğitim, checkpoint varlığı ve başarılı fingerprint kontrolü yapmadan manifest yazıp başarı gösteriyor. `mainwindow.cpp:224–225` okunamayan checkpoint için boş fingerprint döndürüyor.
- **Bağımsız disk kanıtı:** `build/agent-debug/debug/.data/production_model.json:2,7` içinde `demo/patchcore/model.ckpt` ve `image_auroc: 0.978` bulunuyor. Release klasöründeki üretim manifesti de aynı sentetik checkpoint/metriği içeriyor. Proje kökünde ve debug `.data` altında bu demo checkpoint için `Test-Path` sonucu `False`. Bu manifestler 30 Temmuz tarihli eski şemaya ait; mevcut kaynağın yeniden çalıştırılması olarak sunulmuyor.
- **Tetikleyici:** Gerçek eğitim yapılmadan uygulamayı açıp **Set as production model** düğmesine basmak; veya gerçek eğitimden sonra checkpoint silinmişken üretime yükseltmek. Demo seçeneğinin kapalı olması açılış sentetik verilerini kaldırmıyor.
- **Etki:** Geçerli model bulunmadığı halde UI üretim modeli hazır bildiriyor; üretim manifesti sonraki yükleme sırasında başarısız oluyor. Ekrandaki sayılar gerçek model başarımı sanılabiliyor. Mevcut gerçek model başarımı hakkında bu inceleme herhangi bir başarı oranı çıkarmadı.
- **Çözüm:** Result/experiment modeline açık `source=demo|real`, tamamlanma durumu, deney kimliği ve dataset snapshot eklemek; demo sonuçlarını ayrı gösterim durumunda tutmak. Promotion önkoşulları: gerçek tamamlanmış sonuç, mevcut okunabilir checkpoint, boş olmayan SHA-256 ve doğrulanmış manifest bağlantısı. Koşullar sağlanmadığında üretim düğmesini devre dışı bırakmak ve yazma sırasında yeniden doğrulamak.
- **Kabul testi:** Demo kapalı temiz açılışta metrikler boş ve promotion kapalı; demo eğitiminden sonra promotion kapalı; silinen checkpoint başarısızlığında mevcut manifest byte olarak korunuyor; gerçek mevcut checkpoint promotion'ında manifest yol ve fingerprint'i aynı artefakta karşılık geliyor.
- **Yaklaşık efor:** 1–2 geliştirici günü, UI durum modeli ve Qt regresyon kontrolü dahil.

## M02 — İlk promotion sonrasında sonraki kazanan aynı oturumda üretime alınamıyor

- **Tür / önem:** Doğrulanmış hata, **P2**. Statik durum geçişi kanıtı.
- **Kanıt:** `mainwindow.cpp:3104–3105` düğme metnini o anki modele sabitleyip `setEnabled(false)` yapıyor. Dosya genelinde `m_promoteButton` aramasında hiçbir `setEnabled(true)` bulunmuyor. `mainwindow.cpp:2646–2659` yeni eğitim başlatırken ve `mainwindow.cpp:3001–3032` yeni kazananı güncellerken bu durum yenilenmiyor. `mainwindow.cpp:2894–2956` genel iş durumu yöneticisi de promotion durumunu yönetmiyor.
- **Tetikleyici:** Bir modeli üretime alıp uygulamayı kapatmadan yeni bir eğitim/deney tamamlamak.
- **Etki:** Yeni kazanan üretime alınamıyor; düğme önceki modelin üretim metnini göstermeye devam ediyor. Uygulamayı yeniden açma, mevcut sonuçlar da yalnız RAM'de tutulduğundan güvenilir bir iş akışı çözümü değildir.
- **Çözüm:** Promotion durumunu düğmenin yerel bool durumundan çıkarmak; mevcut gerçek sonuç kimliği ile üretim manifestindeki sonuç/artefakt kimliğini karşılaştırarak metin/etkinliği tek `refreshPromotionState()` geçişinde türetmek. Yeni sonuç, kazanan, veri kümesi ve iş durumu değişimlerinde çağırmak.
- **Kabul testi:** A modeli üretime alındıktan sonra B deneyinin kazananı hazır olduğunda düğme B için etkin; B üretime alındığında tekrar pasif; önceki üretim manifesti yeni eğitim sırasında korunuyor ve iş sürerken promotion yapılamıyor.
- **Yaklaşık efor:** 0,5–1 gün.

## M03 — Kalıcı sonuç deposu çalıştırılan EXE klasörüne göre değişiyor

- **Tür / önem:** Koşullu risk, **P2**. Klasör ayrımı mevcut dosya sistemiyle doğrulanmış davranıştır; veri kaybı veya yazma izni hatası bu rol tarafından çalıştırılmadı.
- **Kanıt:** `mainwindow.cpp:3966–3971` veri kökünü `applicationDirPath()/.data` seçiyor ve `mkpath` sonucunu yok sayıyor. Eğitim sonuçları `mainwindow.cpp:2643`, export `3056`, üretim manifesti `3096`, inference-session `3169` ve işler `4090` bu köke bağlı. Model varlıkları ise `mainwindow.cpp:4011–4014` üzerinden keşfedilen proje kökünün `.data/model-assets` dizininde. `main.cpp:8–9` uygulama QSettings kimliğini sabit tutuyor; başka EXE'nin hatırlanan son artefakt yolu farklı build deposuna işaret edebiliyor.
- **Ölçülmüş envanter:** Proje `.data` altında `model-assets`; `build/agent-debug/debug/.data` altında `jobs` ve `production_model.json`; `build/agent-release/release/.data` altında `results/inference`; Desktop release `.data` altında ayrıca `exports`, `results`, `inference`, `production_model.json` bulunuyor. Bu adlar yalnız dizin envanteridir, dosyalar taşınmadı veya silinmedi.
- **Tetikleyici:** Debug'dan release EXE'ye geçmek; EXE'yi başka klasöre dağıtmak; yazma izni olmayan kurulum dizininden çalıştırmak.
- **Etki:** Kullanıcının aynı projeye ait sonuç/üretim modeli birden fazla depoya bölünüyor; yeni EXE önceki üretim manifestini otomatik bulamıyor. Korunan kurulum dizininde worker/iş dosyası hazırlığı da başarısız olabilir. Diskte veri bulunduğu halde arayüzde görünmeme riski var; otomatik veri silinmesi iddiası yok.
- **Çözüm:** Açık ve sabit workspace veri kökü veya `QStandardPaths` tabanlı kullanıcı veri kökü seçmek; sonuç/asset/manifest yollarını tek `WorkspacePaths` hizmetinden türetmek. Mevcut build `.data` depolarını kullanıcıya görünür envanterle tespit eden, eski artefaktları koruyan geçiş tasarlamak; otomatik taşıma/silme yapmamak. Kökün oluşturma/yazma doğrulamasını tek yerde raporlamak.
- **Kabul testi:** Aynı workspace'i debug/release ve taşınmış EXE ile açmak aynı sonuç/manifest deposunu buluyor; salt okunur EXE klasörü kullanıcı veri köküyle iş hazırlamayı engellemiyor; eski depolara erişim/manifest yolları korunuyor.
- **Yaklaşık efor:** 1–2 gün; veri taşıma gerekecekse ek planlama.

## M04 — Python yorumlayıcısı workspace kimliğini değiştirebiliyor

- **Tür / önem:** Koşullu risk, **P2**. Kaynaktan doğrulanan arama önceliği; başka proje kopyası oluşturulup çalıştırılmadı.
- **Kanıt:** `mainwindow.cpp:3993–4007` workspace keşfinde mutlak Python executable dizinini, CWD ve EXE dizininden önce deniyor. `mainwindow.cpp:3978–3984` yalnız `KivancAnomaLibTest.pro` ve `worker/anomalib_worker.py` varlığıyla kökü tanımlıyor. `mainwindow.cpp:2697–2698` seçilen kökü worker'a gönderiyor; `worker/anomalib_worker.py:840–848` göreli ImageNette yolunu bu workspace üzerinden çözüyor. Model asset kökü de aynı keşfe bağlı.
- **Tetikleyici:** Proje A'nın EXE'sinde proje B kopyasının `.venv/Scripts/python.exe` yorumlayıcısını seçmek; ortak veya taşınmış bir Python ortamını proje kökünü belirlemek için kullanmak.
- **Etki:** Model asset durumları, indirme hedefi ve göreli ImageNette yolu B'ye yönelirken sonuç deposu hâlâ A'nın EXE klasöründe kalıyor. B'deki farklı varlık/ayarların yanlış deneyde kullanılma ve pahalı varlıkların gereksiz tekrar hazırlanma riski var. Bu inceleme böyle bir indirme gerçekleştirmedi.
- **Çözüm:** Workspace kimliğini yorumlayıcı seçiminden ayırmak; açık kullanıcı workspace seçimini veya sabit başlangıç proje kökünü tercih etmek. Yorumlayıcı yolu yalnız ortamı belirlemeli. Seçilen workspace/data/asset köklerini teşhis ekranında göstermek ve job snapshot'a yazmak.
- **Kabul testi:** A projesinde B'nin yorumlayıcısı seçildiğinde A'nın workspace ve asset kökü değişmiyor; göreli ImageNette A'ya göre çözülüyor; taşınmış/kurulmuş EXE için seçilen workspace belirli ve kalıcı.
- **Yaklaşık efor:** 0,5–1 gün; M03 ile birlikte ele alınabilir.

## M05 — Görünüm, deney yönetimi ve artefakt yaşam döngüsü MainWindow'da birleşmiş

- **Tür / önem:** İyileştirme fırsatı, **P3**. Büyük dosya tek başına hata ya da ölçülmüş yavaşlık değildir.
- **Kanıt:** `mainwindow.cpp:324` UI kurulumu, `816` parametre formu, `1215` domain doğrulaması, `1435` ayar göçü, `2161` tema, `2582` iş hazırlama, `3062` promotion, `3207/3326` inference oturum durumu ve `3966–4129` dosya/workspace yazımı tek sınıfta. `mainwindow.h:138–140,159,217,227` view nesneleriyle birlikte camera, worker, parametre ve sonuç sahipliği taşıyor. Bu rolün envanter ölçümü: 169.006 bayt / 4.190 satır. Native uygulama içinde ayrı domain/controller sınırı bulunmuyor.
- **Tetikleyici:** Yeni model, yeni deployment formatı veya UI sayfası eklemek; promotion/inference hatasını düzeltmek.
- **Etki:** Görsel değişiklik domain durumuna erişiyor; durum geçişlerinin Qt ekranı açılmadan izole test edilmesi güçleşiyor. M01/M02 gibi eksik geçiş kontrollerinin tekrarlanmasına zemin hazırlıyor; bunun sayısal bakım maliyeti ölçülmedi.
- **Çözüm:** Aşamalı olarak `WorkspacePaths`, typed `ExperimentResult/ModelArtifact`, `TrainingController`, `InferenceController`, `ArtifactRepository` ayırmak. MainWindow yalnız form verilerini iletsin ve durumu göstersin. Önce M01/M02'nin kontrollü durum geçişleri ayrıştırılsın; tüm UI'yı bir seferde yeniden yazmak gerekli değil.
- **Kabul testi:** Promotion önkoşulları ve ikinci deney durum geçişi QWidget açmadan test edilebiliyor; mevcut worker JSON Lines akışı ve UI davranışı korunuyor; controller başka depolama kökü/worker mock'uyla çalışabiliyor.
- **Yaklaşık efor:** İlk kritik ayrıştırma 2–4 gün; tüm sayfalar için 5–10 gün.

## M06 — Model parametre sözleşmesi iki dilde elle çoğaltılıyor

- **Tür / önem:** İyileştirme fırsatı, **P3**. İncelenen varsayılanlarda mevcut C++/Python uyumsuzluğu saptanmadı; öneri olası drift'i önlemek içindir.
- **Kanıt:** C++ varsayılan/model/backbone/layer listeleri `mainwindow.cpp:93–190`, doğrulayıcı `1215–1396`; Python karşılıkları `worker/anomalib_worker.py:26–75,199–219,260–365`. C++ parametre formu aynı aralıkları tekrar tanımlıyor. `mainwindow.cpp:2967–2970` eksik result snapshot'ını güncel varsayılanla dolduruyor; `1424–1432` checkpoint ayarı bulunmadığında yine güncel varsayılanı seçiyor. JSON resource listesi `resources.qrc:3` yalnız worker'ı gömüyor, ortak şema kaynağı yok.
- **Tetikleyici:** Bir parametre varsayılanı/aralığı yalnız worker'da veya yalnız UI'da değiştirilmesi; eski sonuç snapshot'ı yeni sürümle okunması.
- **Etki:** UI'nin kabul ettiği değer worker'da reddedilebilir veya eski artefaktın ayarının varsayım olduğu görünmez kalabilir. README'deki model-map merkezileştirmesi yeni modele UI kartı/parametre formu ekleme işini ortadan kaldırmıyor.
- **Çözüm:** Sürümlü ortak JSON model şemasıyla isim, tür, aralık ve varsayılanları tanımlamak; C++ form/validator ve Python normalizasyonunu bu şemaya bağlamak. Legacy eksik snapshot'a açık `assumed_defaults` provenance eklemek. Alternatif küçük ilk adım: iki tarafın sınır değerlerini aynı fixture'lara karşı contract test ile karşılaştırmak.
- **Kabul testi:** Tüm model varsayılanları, minimum/maximum, yanlış tür ve bilinmeyen anahtar fixture'larında iki taraf aynı kabul/red sonucu veriyor; eski snapshot göçü hangi değerlerin varsayıldığını koruyor; yeni model ekleme ortak şema ve backend factory üzerinden yapılabiliyor.
- **Yaklaşık efor:** Contract test ilk adım 1–2 gün; ortak şema/form üretimi 3–5 gün.

## Uygulama sırası ve sınırlar

Önerilen sıra: M01 → M02 → M03/M04 birlikte → M06 contract test → M05 aşamalı ayrıştırma. Bu not düzeltme uygulamaz.

Gerçekten çalıştırılan kontroller: `rg --files` kaynak/belge envanteri; `rg -n` sembol ve durum mutasyonları; PowerShell `Get-Content` satır numaralı kaynak okuma; kaynak dosya boyutu/satır sayısı; yalnız `.data` üst dizin adlarının envanteri; iki mevcut production manifestinin JSON olarak okunması; iki demo checkpoint yolunun `Test-Path` kontrolü. Hepsi salt okunur kontrollerdir. Bu rol unit test, Qt build, UI smoke test, latency benchmark veya donanım testi yapmadı. GPU/veri modeli başarımı ve SDK çalışma zamanı bu notun kanıt kapsamı dışındadır. Kullanıcı kaynak/veri/model/sonuç dosyaları korundu.
