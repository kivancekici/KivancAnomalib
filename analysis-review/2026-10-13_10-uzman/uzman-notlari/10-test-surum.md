# Uzman 10 — Testler, kurulum, tekrar üretilebilirlik ve sürümleme

İnceleme tarihi: 13 Ekim 2026. Rol bağımsız kaynak incelemesi, mevcut test kayıtlarının değerlendirmesi, yerel paket metadata kontrolü ve küçük resource/hash probe'u ile tamamlandı. Kaynak/konfigürasyon değiştirilmedi; paket yükleme, eğitim, üretim modeli yükleme, kamera, GUI, native build, indirme veya upload yapılmadı.

## Mevcut durum ve kanıt

- Kapsam: `tests/test_worker.py`, `tests/demo_job.json`, `README.md`, `requirements-inference.txt`, `KivancAnomaLibTest.pro`, `resources.qrc`, `main.cpp`, `mainwindow.cpp:4017–4178`, worker sürüm/protokol/environment bölümleri ve küçük build/resource metadata. Git deposu veya AGENTS talimatı bulunduğu varsayılmadı.
- Ana ajanın [test kaydı](../kanitlar/unittest-workspace-temp.log), **37 test / 67,930 saniye / OK** bildiriyor. [Komut sonucu](../kanitlar/unittest-workspace-temp-sonuc.json) exit 0 ve toplam süreç süresi 70,347 saniye. Bu rol tüm suite'i tekrar çalıştırmadı. İlk varsayılan TEMP denemesindeki 29 PermissionError, review içindeki TEMP/TMP ile giderilmiş sandbox kısıtı; uygulama hatası değildir.
- Bu rolün [kaynak/resource/test inventory probe'u](../kanitlar/10-surum-probe.py), `.venv/Scripts/python.exe -B -X utf8` ile **exit 0** (yaklaşık 0,8 saniye). Yalnız kaynak ve üretilmiş C++ resource metni okundu; worker resource baytları zlib ile bellekte açıldı. [JSON sonucu](../kanitlar/10-surum-probe.json) 37 test tanımını ve hash'leri saklar.
- Salt okunur `.venv/Scripts/python.exe -B -X utf8 -m pip check` **exit 0** (yaklaşık 2,95 saniye), [log](../kanitlar/10-pip-check.log): `No broken requirements found.` Bu, paket metadata bağımlılıklarını doğrular; native DLL, GPU çalışma ve model kalitesi sonucu değildir.
- `PYTHONDONTWRITEBYTECODE=1`, `-B`, TEMP/TMP=`kanitlar/test-temp` kullanıldı. Yeni çıktılar yalnız inceleme klasörüne yazıldı. [Kurulu sürümler](../kanitlar/ortam-surumleri.json): Python 3.12.14, Anomalib 2.6.0, Torch 2.13.0+cu130, Torchvision 0.28.0+cu130, Lightning 2.6.5, timm 1.0.28, OpenVINO 2026.2.1, ONNX 1.22.0, ORT GPU 1.28.0. `.venv/pyvenv.cfg:3` Python 3.12.13 ve Codex cache'deki temel interpreter'ı gösteriyor; çalışan interpreter'ın patch sürümü 3.12.14. Bunun tek başına bir Python uyumsuzluğu olduğu iddia edilmez.

## Öncelikli bulgular

### T10-01 — Tam ve doğrulanmış kurulum manifesti bulunmuyor

**Tür / önem:** Koşullu risk, **P2**; mevcut ortamda bozuk paket bağımlılığı bulunmadı.

**Kanıt:** [README:63](../../../README.md) kullanıcıya Anomalib sürümünü seçip kurmasını söylüyor; yalnız [requirements-inference.txt:3](../../../requirements-inference.txt) `onnxruntime-gpu>=1.27,<1.30` içeriyor. Proje kaynaklarında tam requirements/lock/pyproject kurulum manifesti bulunmadı. [Worker:146](../../../worker/anomalib_worker.py) yalnız Anomalib major/minor bandını denetliyor; [test:709](../../../tests/test_worker.py) `2.9.0rc1` değerini açıkça kabul ediyor. Bu geniş API bandının tüm sürümleriyle doğrulanmış eğitim/export uyumu gösterilmiyor. Native [qmake projesi:23](../../../KivancAnomaLibTest.pro) Pylon/OpenCV yollarını ve `v12`/`4120` kütüphane adlarını makineye bağlı seçiyor. [Python seçimi:4171](../../../mainwindow.cpp) proje venv'i yoksa makinedeki Codex cache Python'una, sonra PATH Python'una düşebiliyor.

**Tetikleyici:** Yeni makine kurulumu, `.venv` kaybı veya kitaplıkları güncelleme; aynı job JSON'u farklı Python/Torch/Anomalib/DLL bileşimiyle çalıştırma.

**Etki:** Başarılı test edilen ortam yeniden kurulamıyor; aynı parametreler farklı API, scheduler, preprocessing veya runtime davranışı üretebilir. EfficientAD scheduler'ın mevcut 2.6.0 davranışı uzman03 **03-04** ile somut sürüm hassasiyeti örneğidir. Burada bağımlılık bandının bugün kesin bozuk olduğu veya CUDA13 gereksiniminin yanlış olduğu öne sürülmez.

**Çözüm:** Desteklenen Windows/Python/CPU ve CUDA ortamlarını açık manifestlerle tanımlamak; üst ve transitif paket sürümlerini, CUDA wheel kaynağını ve hash'lerini kilitlemek; Qt/MSVC/Pylon/OpenCV sürüm ve DLL dağıtım listesini kaydetmek. Gerçek çalıştırılan interpreter, paket sürümleri ve ortam manifest hash'ini run/export kaydına eklemek. Geniş Anomalib bandını yalnız doğrulanmış bir uyum matrisi varsa ilan etmek. Resmi [pip tekrarlanabilir kurulum belgesi](https://pip.pypa.io/en/stable/topics/repeatable-installs/) tam sürüm pinlerini, transitif bağımlılıkları ve hash kontrolünü açıklar; platforma bağlı wheelhouse ayrıca paketlenebilir.

**Kabul testi:** Temiz, desteklenen ortam yalnız kaydedilmiş manifestten kurulabilmeli; `pip check`, worker suite ve küçük gerçek CPU graph sözleşme kontrolleri geçmeli. Seçilmiş CPU/CUDA profilleri ayrı raporlanmalı. Job kaydı farklı interpreter/manifest hash'ini açıkça ayırt etmeli. Bu incelemede temiz kurulum yapılmadı.

**Efor:** İlk doğrulanmış manifest ve kurulum yönergesi 1 gün; temiz makine/CPU–CUDA kabul matrisi 1–2 gün ek çalışma.

### T10-02 — Kaynak testleri ile paketlenen worker'ın kimliği bağlanmıyor

**Tür / önem:** Koşullu risk, **P2**. Eski resource/kopya farkları doğrulanmış; mevcut EXE'nin yanlış worker kullandığı doğrulanmamıştır.

**Kanıt:** [tests:21](../../../tests/test_worker.py) worker'ı kaynak dizininden çalıştırıyor. [resources.qrc:3](../../../resources.qrc) aynı dosyayı binary resource'a alıyor; [ensureWorkerScript:4024](../../../mainwindow.cpp) çalışırken worker'ı kaynak dizininden değil **EXE resource'undan** çıkarıyor. [4047](../../../mainwindow.cpp) disktekini embedded içerikle karşılaştırıp atomik yeniliyor; bu iyi davranış, fakat eski binary güncel kaynak worker'ını otomatik almaz. [main.cpp:10](../../../main.cpp) sürüm `0.1.0`; [session_ready:3328](../../../worker/anomalib_worker.py) yalnız PID yayımlıyor; build/source/protocol kimliği yok.

**Bağımsız ölçüm:** [Resource/hash probe sonucu](../kanitlar/10-surum-probe.json) şu üretilmiş resource worker'larını doğruladı:

| Kapsam | Açılan worker boyutu | SHA-256 başlangıcı | Güncel kaynakla aynı mı? |
| --- | ---: | --- | --- |
| Kaynak `worker/anomalib_worker.py` | 130118 | `bf9fe0b1a3929909` | Evet |
| `build/agent-debug/debug/qrc_resources.cpp:15` | 126272 | `df1c5875dee99219` | Hayır |
| `build/agent-release/release/qrc_resources.cpp:15` | 120088 | `e41b3189ad693bd1` | Hayır |
| `build/Desktop_Qt_6_11_1_MSVC2022_64bit_Debug/debug/qrc_resources.cpp:15` | 17372 | `ab8f30d75cecc56e` | Hayır |
| `build/Desktop_Qt_6_11_1_MSVC2022_64bit_Release/release/qrc_resources.cpp:15` | 130118 | `bf9fe0b1a3929909` | Evet |

Disk `.data` kopyaları agent-debug'da 35845 bayt/`7f16b11ea821f46a`, agent-release'de 120088 bayt/`e41b3189ad693bd1`, ana Release'de 130118 bayt/`bf9fe0b1a3929909`. Agent-debug resource ile cache kopyası da birbirinden farklı; atomik yenileme kaynakta bu durumu çözmeye uygundur. Üretilmiş C++ resource dosyası EXE'nin gerçekten bunlarla linklendiğinin kanıtı değildir; EXE açılmadı, çalıştırılmadı ve embedded içeriği doğrudan okunmadı.

**Tetikleyici:** Kaynak worker güncellendikten sonra eski build/EXE'yi kullanma veya farklı build klasöründeki executable'ı aynı uygulama sürümü adıyla dağıtma.

**Etki:** Kaynaktaki 37 testin geçtiği sürüm ile operatörün çalıştırdığı worker/protokol farklı olabilir; hata raporunun hangi sürümden geldiği belirlenemez. Eski build'ler normal tarihsel artefakt olabilir; silinmeleri veya yeniden build edilmeleri bu analizde yapılmadı.

**Çözüm:** Build manifestine worker SHA, native kaynak manifest SHA, uygulama build kimliği ve JSON protokol sürümü koymak; handshake ve sonuçlarda bunları yayımlamak. Paket kabulünde binary resource worker'ı ile test edilen kaynak hash'ini karşılaştırmak; tek tanımlı dağıtım çıktısı ve saklanan tarihsel build envanteri. Qt [6.11 resource belgesi](https://doc.qt.io/qt-6.11/resources.html), qmake `RESOURCES` içeriğinin build sırasında executable'a katıldığını doğrular.

**Kabul testi:** Yalnız worker'ı değiştirip native paketlemeyi güncellemeden hazırlanan aday, release kabulünde hash uyuşmazlığıyla reddedilmeli. Güncel paket çıkarılan worker hash'ini test raporundaki hash'le eşleştirmeli; handshake bilinmeyen protokol sürümünü açıkça bildirmeli. Release/debug build kimlikleri ve karşılık gelen test kayıtları ayırt edilebilmeli.

**Efor:** Kimlik/manifest/handshake 0,5–1 gün; paket kabul kontrolü 0,5–1 gün.

### T10-03 — Test başarısı doğrulanan karar ve export sözleşmelerini henüz kapsamıyor

**Tür / önem:** İyileştirme fırsatı, **P2**. Bu başlık yeni bir model hatası sayılmamalı; diğer uzmanların somut hatalarını kalıcı regresyon kapsamına bağlar.

**Kanıt:** [tests:166](../../../tests/test_worker.py) gerçek ORT CPU Gaussian blur parity'sini kontrol ediyor, ancak graph validation kalibrasyon normalizasyonu içermiyor. [tests:279](../../../tests/test_worker.py) resize backend'lerini `__new__`/fake çalıştırıyor; gerçek metadata'sız graph shape/type kontrolü yok. [tests:315](../../../tests/test_worker.py) session'ı FakeBackend ile tahmin ediyor. [tests:717](../../../tests/test_worker.py) yalnız demo export manifestini inceliyor. [tests:679](../../../tests/test_worker.py) üç modelin gerçek constructor ve rectangular preprocessor'ını kontrol etmesi yararlı; bütün testlerin mock olduğu söylenemez.

**Tetikleyici:** Piksel eşiği/karar adaptörü veya ONNX export/backend değişikliği sonrasında bu suite'i tek kabul ölçütü saymak.

**Etki:** Suite geçerken uzman04 **M04-01** (görüntü içi min–max/eşik), **M04-02/03** (eksik/NaN skor/metrik), uzman06 **D06-01** (export üzerine yazma), **D06-02** (kalibrasyon), **D06-03/04/05** (external-data/input/output sözleşmesi) sürüyor. Bu bağ, test açığını somut sonuçla ilişkilendirir; başarı oranı veya gerçek model kalitesi çıkarımı yoktur.

**Çözüm:** Bu uzmanların yeni küçük ve çevrimdışı bağımsız probe'larını gerekli düzeltmeler sırasında outcome testlerine dönüştürmek: kalibre 0,2/0,4 haritada eşik 0,5'in mask sonucu; negatif/NaN/eksik skorun karar/JSON etkisi; ikinci export sonrası ilk manifest/artefaktın hâlâ geçerli olması; external weight mutasyonu/eksikliği; sabit graph input shape/dtype ve eksik output'un load aşamasında açık reddi. Test beklenen değeri uygulama fonksiyonundan üretmemeli. Hızlı CPU sözleşme testleri ile sonra izin verilen gerçek model/hardware kabulünü ayrı koşular halinde tutmak.

**Kabul testi:** Düzeltilmemiş mevcut kaynak üzerinde her yeni test ilgili somut nedenle başarısız olmalı; düzeltme sonrasında geçmeli. Var olan 37 test korunmalı. Şema/karar output'u bağımsız sabit/numerik oraklla doğrulanmalı; gerçek backend CPU tiny graph'ları fake constructor gerektirmeden kullanılmalı. Export kabulü gerçek büyük model/eğitim gerektirmeden path/hash değişmezliğini sınamalı.

**Efor:** İlk 5–8 öncelikli küçük regresyon testi 1–2 gün; native süreç/UI ve gerçek model parity kabulü ayrı kapsam.

### T10-04 — “CUDA provider ready” doğrulaması gerçek yürütmeyi sınamıyor

**Tür / önem:** Koşullu risk, **P2**; provider-listesiyle hazır beyanı üretildiği doğrulanmış, mevcut GPU'nun çalışmadığına ilişkin kanıt yoktur.

**Kanıt:** [run_check:3379](../../../worker/anomalib_worker.py) `ort.get_available_providers()` listesini, [3387](../../../worker/anomalib_worker.py) OpenVINO cihaz adlarını sorguluyor. [3396](../../../worker/anomalib_worker.py) listede CUDA varsa `ONNX CUDA provider ready`; [3403](../../../worker/anomalib_worker.py) `ok=True` yayımlıyor. Session kurulumu, native DLL preload, graph compile/run veya çıktının kontrolü bu yolda yok. Gerçek ORT backend [2893](../../../worker/anomalib_worker.py) ayrıca DLL preload ve [2911](../../../worker/anomalib_worker.py) Session oluşturuyor. [test:888](../../../tests/test_worker.py) `ok`/`message` anahtarlarının varlığını sınar; hem eksik hem hazır ortamı destekleme amacıyla `ok=True` veya execution beklememesi doğrudur, ancak deployment kabulü değildir.

**Tetikleyici:** ORT paketinin CUDA provider'ını ilan ettiği, gerekli native DLL/driver/graph kernel'inin ilk Session/run sırasında sorun çıkardığı bir makine.

**Etki:** Operatörün yeşil ortam mesajını GPU inference onayı sayması; gerçek model yüklerken fallback/hata. Ana ajanın [ortam kontrolü](../kanitlar/ortam-check.log) bu makinede liste/cihaz keşfinin başarılı olduğunu gösterir; gerçek GPU execution kanıtı değildir. Resmi [ORT Python API](https://onnxruntime.ai/docs/api/python/api_summary.html), Session oluşturma/graph yürütme ile provider tercih/fallback davranışlarını ayrı tanımlar.

**Çözüm:** Durumları `detected`, `runtime_smoke_passed`, `model_loaded` ve gerçek model output kabulü olarak açık ayırmak; mevcut kontroldeki mesajı keşif kapsamına uygun yazmak. İstenirse açık ayrı verification adımı, indirmeden birkaç KB'lık deterministik graph'ı istenen provider ile çalıştırıp Session provider ve node yürütme kanıtını raporlamalı; CPU fallback GPU başarısı sayılmamalı. Üretim modeli parity'si bundan ayrı olmalı.

**Kabul testi:** Listede CUDA varken Session/run hata verince “GPU smoke başarılı” durumu üretilmemeli. CPU fallback ayrı raporlanmalı. CPU/CUDA tiny graph aynı sabit output'u tolerans içinde vermeli; gerçek kullanılmış provider kanıtı saklanmalı. Bu incelemede GPU smoke çalıştırılmadı.

**Efor:** Mesaj/durum ayrımı 0,5 gün; küçük ayrı runtime-smoke ve failure/fallback testleri 1 gün.

### T10-05 — README doğrulama kayıtları güncel kapsamı yanlış tarif ediyor

**Tür / önem:** Doğrulanmış dokümantasyon hatası, **P3**. Geçmiş native/kamera smoke beyanlarının yanlış olduğu ileri sürülmez.

**Kanıt:** [README:217](../../../README.md) 26 unit test yazarken kaynak ve mevcut çalıştırılmış kayıt 37 gösteriyor. [README:84](../../../README.md) persistent süreç/ONNX/OpenVINO yüklemeyi anlatırken [224](../../../README.md) one-shot ve continuous exported-model deployment'ı gelecek artım olarak tanımlıyor. Worker `--serve`, kalıcı backend/tekrar kullanım mevcut. Buradaki “continuous” gerçek sürekli kamera akışını kastediyor olabilir; **sürekli kamera incelemesi mevcut diye yorumlanmamalı**, yalnız persistent backend ile one-shot iş ayrımı belirsiz. Native/kamera “passed” satırları build/hash/tarih/test kaydına bağlanmamış.

**Tetikleyici:** Yeni geliştirici/operatörün README test bölümünü güncel release kabul kaydı veya mevcut Inspect mimarisinin kesin tanımı sayması.

**Etki:** Çalıştırılmış test sayısı, binary sürümü, statik/sentetik kontrol ve tarihsel donanım kabulü karışır; doğrulama eksikliği fark edilmez.

**Çözüm:** Test sayısını güncel suite'ten üretmek veya sayıyı sabitlemeden komut/kapsam yazmak; her release için build/worker hash, tarih, ortam, gerçek command/result ve hardware erişimini kaydetmek. Persistent model süreci, tek görüntü request'i ve henüz ayrı kapsam olan sürekli kamera loop'unu net tariflemek. Uzman09 UI/belge bulgusuyla birleştirilmeli; bağımsız teknik düzeltme olarak iki kez sayılmamalı.

**Kabul testi:** README test komutu/tanımı mevcut 37 testle tutarlı olmalı; release kaydı doğru worker/build kimliğine bağlanmalı. Native/kamera geçmiş testleri bu analizde yeniden çalıştırılmış gibi görünmemeli; `--serve`/persistent kullanım ile gerçek sürekli kamera desteği ayrı anlaşılmalı.

**Efor:** 0,5 gün.

## Sonuç ve uygulama sırası

Önce kritik karar/export düzeltmeleriyle birlikte T10-03 regresyon kapsamı; ardından T10-02 paket kimliği ve T10-01 doğrulanmış ortam manifesti; T10-04 keşif/yürütme ayrımı; T10-05 belge ve release kayıtları. Seed/veri ayrımı uzman02, etkin trainer parametre kayıtları uzman03, asset staging/dayanıklılık uzman08, artefakt sözleşmeleri uzman06 tarafından ele alınır; aynı konular burada yeni bağımsız model hataları olarak sayılmaz.

37 testin geçmesi, mevcut worker'ın kapsanan demo/constructor/artifact/asset/protokol davranışları için iyi kanıttır. Gerçek veri üzerindeki anomalib doğruluk/yanlış alarm/kaçırma oranı, mevcut EXE'nin embedded içeriği, kamera davranışı, uzun eğitim, büyük export ve üretim GPU/ORT/OpenVINO parity'si bu rol kapsamında doğrulanmadı.
