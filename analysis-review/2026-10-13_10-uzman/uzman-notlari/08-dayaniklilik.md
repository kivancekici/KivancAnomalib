# Uzman 08 — Hata yönetimi, kaynak ömrü ve dayanıklılık

13 Ekim 2026, Europe/Istanbul. Ayrı yapay zekâ uzman incelemesi tamamlandı. Uygulama kaynakları ve konfigürasyon değiştirilmedi; kamera, GUI, gerçek model/inference/eğitim, paket kurulumu ve veri/model indirme çalıştırılmadı.

## Kapsam, yöntem ve karar

`anomalibprocess.cpp/.h` ve `cameraservice.cpp/.h` tümü; `mainwindow.cpp` constructor/destructor/close, training start/stop/state, kalıcı inference ve capture yolları; `worker/anomalib_worker.py` varlık staging/promotion, artefakt yazımı, kalıcı oturum ve main/serve okundu. Ana envanterin talimat aramasıyla uyumlu olarak uygulanabilir AGENTS.md bulunmadı. README'deki güvence ve eski kayıtlar kanıt olarak değerlendirildi.

İlk işler varlık promotion/geri alma güvenliği ve inference istek/durum korelasyonu olmalı. Python servis döngüsü bozuk komutları izole edebiliyor; buna karşılık dosya promotion başarısızlığı önceki managed klasörü korumuyor. UI ve süreç kapanışı için aşağıdaki riskler kaynak koddan çıkarılmıştır; gerçek crash, kilitlenme veya donanım gecikmesi ölçüldüğü iddia edilmez.

Kaynakta QtConcurrent, QFuture, QFutureWatcher veya `std::thread` kullanımı yoktur. Kamera nesnesinin bu tür bir lambda ile kapanıştan sonra kullanılacağına ilişkin bulgu yoktur. Kamera çağrısı UI slotundan senkron yapılır. Python işi QProcess ile ayrı süreçtedir. `Pylon::CInstantCamera`, grab-result ve dönüştürülen görüntü yerel RAII nesneleridir; `cv::Mat` ödünç buffer'ı dönüştürülen görüntü hayattayken `imwrite` içinde kullanılır. Bu kapsamda doğrulanmış buffer use-after-free saptanmadı.

## Gerçekte çalıştırılan kontroller

`.venv/Scripts/python.exe -B -X utf8 analysis-review/2026-10-13_10-uzman/kanitlar/08-dayaniklilik-probe.py` exit 0; araç duvar süresi 1,3 saniye. `PYTHONDONTWRITEBYTECODE=1`, inceleme altındaki TEMP/TMP kullanıldı. Bu süre uygulama/model performans ölçümü değildir. Sentetik dosyalar yalnız probe'un benzersiz review alt klasöründe oluşturuldu; silme/taşıma hedefleri önceden bu klasör altında doğrulandı. Üretim varlıklarına dokunulmadı. Kaynak: [probe](../kanitlar/08-dayaniklilik-probe.py), sonuç: [JSON](../kanitlar/08-dayaniklilik-probe.json).

- Rename hatası enjekte edildi: eski destination artık yok, staging hâlâ var.
- HF import sonrası format doğrulaması hata verdi: önceki sentinel kaybolmuş, hatalı aday managed klasöre taşınmış.
- `predict` komutunda `output_root` verilmedi: `.` kabul edildi. Backend ve artefakt yazıcısı mock'tur; çalışma klasörüne dosya yazılmadı.
- Olumlu servis kontrolü: bozuk JSON ve bilinmeyen komuttan sonra olay sırası `session_ready → error → error → session_stopped`, çıkış 0; bilinmeyen komutun `request_id` alanı doğru korunuyor. Backend, offline yapılandırma ve network guard mock'landı; asıl `run_inference_service` döngüsü çalıştı.
- Yerel `C:/Qt/6.11.1/msvc2022_64/bin/qmake.exe -query QT_VERSION` çıktısı 6.11.1. Basler SDK `PylonVersionNumber.h:9–12` sürümü 12.2.0.1265. Yerel `InstantCamera.h` bekleyen RetrieveResult davranışını da belgeliyor. Qt'nin resmi 6.11 belgesi webde 6.11.2 başlığıyla sunuluyor; burada kullanılan API sözleşmeleri aynı 6.11 serisindedir.

Ana koordinatörün mevcut 37 test / 67,930 saniye OK sonucu bu rol tarafından yeniden çalıştırılmadı. Python mock kontrolü native Qt sinyal zamanlamasını, Windows kill sonrasındaki gerçek gecikmeyi veya kamera davranışını sınamaz.

## Bulgular

### 08-01 — Varlık promotion başarısızlığında önceki managed klasör kayboluyor

**Tür / önem:** Doğrulanmış hata, P2. Managed varlık kaybıdır; kullanıcı veri kümesinin veya özgün HF kullanıcı cache'inin silindiği iddiası değildir.

**Kanıt:** `worker/anomalib_worker.py:1012–1016` önce `shutil.rmtree(destination)` yapıyor, sonra `staging.replace(destination)` çağırıyor; rollback yok. `_import_hf_asset` promotion'ı `1043`'te yapıyor, safetensors/sha256 doğrulamasını ancak `1050–1058`'de uyguluyor. Archive yolu `1252`'de aynı helper'ı kullanıyor. Sentetik PermissionError kontrolünde eski destination silindi; import format kontrolü başarısızlığında hatalı aday managed konuma çoktan taşınmıştı. Asset-lock dosyasının geçici dosya + replace ile yazılması (`439–447`) olumlu bir koruma olsa da klasör işlemini atomik kılmaz.

**Tetikleyici:** Destination silindikten sonra rename'in erişim/antivirüs/disk hatası vermesi; import edilen adayın validation'da başarısız olması; tam bu aralıkta zorunlu süreç sonlandırma. Normal akış mevcut varlığı `ready` ise atlar (`1294–1303`); kusur mevcut fakat eksik/yanlış lock'lı/corrupt olarak işaretli klasörün yenilenmesinde önemlidir.

**Etki:** Önceki managed dosyalar ve diğer sağlam cache içerikleri kaybolabilir, kayıt ile disk çelişebilir; kullanıcı yeniden hazırlama/indirme yapmak zorunda kalabilir. Disk/rename arızası gerçek ortamda oluşturulmadı.

**Çözüm:** Adayı staging içinde tamamen doğrulamak; versioned varlık klasörü ve tamamlanmış sürüme işaret eden manifest kullanmak veya önceki klasörü backup'a taşıyıp başarısız promotion'da rollback yapmak. Önceki kopyayı yeni varlık ve lock başarıyla tamamlanmadan silmemek; aynı asset root için süreçler arası kilit eklemek. Windows'ta dolu klasör değiştirme için tek dosyalık atomic replace güvencesi varsayılmamalı.

**Kabul testi:** Promotion öncesi format/sha256 failure, rename failure ve lock-write failure enjekte edildiğinde önceki dosyalar/kayıt kullanılabilir kalır; restart tamamlanmamış staging'i başarı saymaz. İki eşzamanlı hazırlama kontrollü kilit altında çakışmaz.

**Efor:** 1–2 geliştirici günü.

### 08-02 — Birden fazla model yükleme isteğinde UI hazır bilgisi yanlış modele bağlanabiliyor

**Tür / önem:** Koşullu risk, P2; geçiş kodu statik olarak doğrulandı, native GUI yarışı çalıştırılmadı.

**Kanıt:** `mainwindow.cpp:3207–3234` pending path ve tek `m_inferenceHadReadyModelBeforeLoad` bayrağıyla istek yolluyor. `setInferenceModelState` (`3121–3160`) model yükleme düğmelerini kapatmıyor; bu düğmeler yalnız ayrı training/export işinin `setJobRunning` durumunda kapanıyor (`2933–2937`). `handleInferenceEvent` model_loaded/inference olayında `request_id` eşleştirmiyor; herhangi bir model_loaded olayı pending bilgiyi ve had-ready bayrağını temizliyor (`3361–3373`). Hatalar yalnız request-id önekiyle değerlendirilir (`3444–3453`). Worker komutları seri işlese de model değişimi başarısızlığında önceki backend'i korur (`worker/anomalib_worker.py:3193–3222`).

**Tetikleyici:** A hazırken B yüklemesini başlatmak; B tamamlanmadan C için ikinci yükleme göndermek; B başarılı, C başarısız olsun. B'nin olayı global bayrağı sıfırlar. C hatası UI'yi `ready=false` yapar, oysa worker'da B hâlâ yüklüdür. B tamamlanınca C hâlâ yüklenirken test düğmesi de kısa süreyle etkinleşebilir. Büyük model yükleme süresi bu olay sırasını mümkün kılar; burada gerçek süre ölçülmedi.

**Etki:** Geçerli backend varken operatör test yapamaz veya görünen model/durum devam eden isteği açıklamaz; gereksiz yeniden yükleme ve tanılama güçlüğü oluşur. İstekler server'da paralel inference yapmıyor; risk UI'nin seri olayları yanlış durumla ilişkilendirmesidir.

**Çözüm:** Aktif load/predict request ID ve session-generation saklamak; yalnız ilgili terminal olayın pending durumunu kapatmasını sağlamak. Tek yükleme sürerken model değişimini kapatmak ya da bilinçli bir sıra yönetimi eklemek. Worker hata olayında `loaded_model_available` ve aktif fingerprint bilgisini döndürsün; UI global had-ready tahminine dayanmasın. `sendCommand` dönüşü (`3234`, `3592`) kontrol edilerek gönderilemeyen istek bekleyen duruma girmesin.

**Kabul testi:** Mock worker ile A hazır → B başladı → C kuyruğa girdi → B loaded → C error dizisinde B kullanılabilir kalır; eski session/request yanıtı yeni isteğin pending durumunu temizlemez. IPC write failure sonucunda düğmeler beklemede kalmaz.

**Efor:** 1–2 gün.

### 08-03 — Durdurma, süreç bitişi ve UI'nin terminal durumu tek sözleşmeyle yönetilmiyor

**Tür / önem:** Koşullu risk, P2; gerçek sızan süreç veya crash doğrulanmadı.

**Kanıt:** `anomalibprocess.cpp:143–166` shutdown için bekliyor; süre aşımında terminate, sonra kill uyguluyor, kill sonrasında bitiş onayı beklemiyor. `mainwindow.cpp:2664–2667` stop dönüşünde UI'yi doğrudan Stopped yapıyor. `processError` tüm hata türlerini aynı jobFailed olayına çeviriyor (`193–196`); `handleWorkerFailure` aktif iş bilgisini temizleyip kontrolleri açıyor (`2879–2891`). Başlangıç `waitForStarted(3000)` false dönerse hata bildiriliyor (`anomalibprocess.cpp:44–49`, `122–126`), ama süreç Starting kaldığında ayrıca durdurulmuyor.

**Resmi davranış:** Qt 6.11'de `Timedout` süreç durumunu değiştirmez; `WriteError`, çocuk stdin'ini kapattığında da oluşabilir. Windows `terminate` WM_CLOSE yollar, konsol süreci bunu işlemeyebilir; `kill` TerminateProcess kullanır. QProcess destructor süreci sonlandırıp bitişi bekler; bu yüzden yalnız stop kodundan kalıcı orphan sonucu çıkarılamaz. [Qt 6.11 QProcess](https://doc.qt.io/qt-6.11/qprocess.html).

**Tetikleyici:** Uzun model load/predict sırasında shutdown'ın sırada beklemesi (`worker/anomalib_worker.py:3329–3354`); 3 saniyeyi aşan process başlangıcı; çalışan süreçte stdin/write failure. Son kill sonrasında UI'nin terminal sinyali almadan yeni işi etkinleştirmesi olasıdır.

**Etki:** “Stopped/Worker unavailable/Job failed” durumu sinyal sırasına göre değişebilir; backend hâlâ geçişteyken kontroller açılır veya session başlatılmış görünmesine rağmen komut gönderilemez. Zorunlu sonlandırma Python finally temizliğini garanti etmez; 08-01 gibi staging geçişleri ve doğrudan PNG yazımları (`1867–1896`) tamamlanmamış kalabilir.

**Çözüm:** Starting/Running/Stopping/Stopped/Failed durumlarını açık yönetmek; terminal UI geçişini finished veya doğrulanmış NotRunning ile yapmak; cancellation'ı normal iş başarısızlığından ayırmak. `FailedToStart/Crashed` ile kanal hatalarını ayrı ele almak; start timeout sonrası çalışan/başlayan süreci kontrollü kapatmak. Önce kooperatif iptal ve sınırlı shutdown timeout, sonra zorunlu stop; bitiş onayını event loop üzerinden almak. Restart yarım staging/artefaktları başarı saymasın.

**Kabul testi:** Gerçek model kullanmayan test child süreçleriyle yavaş startup, stdin-close, uzun command ve shutdown-yanıtsız durumlarında kontroller yalnız gerçek bitişte açılır; iptal edilen iş “tamamlandı” olmaz; uygulama kapanışında iki worker'ın terminal durumu doğrulanır.

**Efor:** 1–2 gün.

### 08-04 — Eksik output_root çalışma klasörüne artefakt yazımını etkinleştiriyor

**Tür / önem:** Doğrulanmış protokol doğrulama hatası, P2; normal Qt UI isteğinde alan mevcut olduğundan o yolda tetiklenmez.

**Kanıt:** `worker/anomalib_worker.py:3257–3259` önce boş değerden `Path('')` oluşturuyor. Bu yolun string değeri `.` olduğundan `if not str(output_root)` asla boş değer denetimi yapmaz. Probe eksik alanı kabul edip writer'a `.` gönderdiğini doğruladı; gerçek writer mock'landı. `write_prediction_artifacts` (`1877–1880`) bu konumda sabit isimli üç PNG oluşturur. UI normal komutuna `output_root` ekliyor (`mainwindow.cpp:3581–3587`).

**Tetikleyici:** Elle/gelecekteki bir istemciyle gönderilen eksik veya boş output_root içeren predict komutu. Göreli yol da doğrulama olmadan kabul edilir.

**Etki:** Çalışma klasöründe beklenmeyen PNG'ler ve aynı adlı eski dosyalarda üzerine yazma riski; çıktı yolu hatası erken anlaşılmaz. Bu incelemede gerçek çalışma klasörüne yazılmadı.

**Çözüm:** Ham alanı Path'e çevirmeden önce string, trim sonrası nonempty ve istenen sözleşmeye göre absolute path olarak doğrulamak. Her request için benzersiz output dizini yaratmak; tamamlanmış artefakt manifestini ancak tüm dosyalar yazılınca yayımlamak.

**Kabul testi:** Eksik/boş/whitespace output_root predict başlamadan ValueError döndürür, backend çağrılmaz ve hiçbir artefakt oluşmaz; geçerli absolute review dizini kabul edilir. Göreli yol politikası açık ve test edilmiştir.

**Efor:** 0,25–0,5 gün; artefakt completion manifesti 0,5 gün ek.

### 08-05 — Kamera bekleme ve süreç shutdown beklemeleri UI olay döngüsünü bloke edebilir

**Tür / önem:** Koşullu risk, P2; gerçek donanım gecikmesi veya belirli FPS ölçülmedi.

**Kanıt:** Kamera capture doğrudan UI slotlarında çağrılır (`mainwindow.cpp:3612`, `3874`, `3895`); kamera `RetrieveResult(5000)` ile bekler (`cameraservice.cpp:86`). Başlangıç/bitiş wait çağrıları da UI'deki start/stop/close yollarından çağrılır (`anomalibprocess.cpp:46`, `124`, `155–163`; `mainwindow.cpp:278–288`). Basler API sonuç yoksa beklediğini ve timeout'un milisaniye olduğunu belirtir. [Basler CInstantCamera resmi dokümanı](https://docs.baslerweb.com/pylonapi/cpp/class_pylon_1_1_c_instant_camera). Qt senkron wait çağrılarının ana thread UI'sini dondurabileceğini belgeler. [Qt 6.11 senkron süreç API'si](https://doc.qt.io/qt-6.11/qprocess.html#synchronous-process-api).

**Tetikleyici:** Kamera kare üretmiyor/trigger bekliyor; model load veya predict 1,5 saniyelik shutdown aralığından uzun sürüyor; Windows konsol süreci terminate'e yanıt vermiyor. Tek camera RetrieveResult için 5 saniye, session shutdown için 0,5 + 1,5 + 2 saniye wait bütçesi kodda vardır; bunlar gerçek normal latency değerleri veya tam fonksiyon süre üst sınırları değildir.

**Etki:** İptal/kapatma ve worker olaylarının görünür işlenmesi gecikir; operatör uygulamanın donduğunu düşünebilir. Kamera/GUI açılmadığından gerçekleşmiş kilitlenme iddiası yoktur.

**Çözüm:** Kamera görevini açık sahiplik/iptal modeliyle worker thread'e taşımak; SDK ömrünün görevi kapsamasını, pencere kapanırken görevin tamamlanmasını sağlamak. QProcess başlangıç/kapanışını started/finished/stateChanged ve timer tabanlı asenkron akışa geçirmek. UI'de capturing/stopping durumu ve iptal sonucu göstermek.

**Kabul testi:** Kamera SDK'sı yerine 5 saniye bekleyen fake servis ve uzun command'lı küçük child süreç kullanırken UI heartbeat/test timer çalışmaya devam eder; pencere kapanışı tamamlanmadan kamera görevi durur; nesne ömrü sanitizer/deterministik task gate ile doğrulanır. Gerçek kamera testi sonraki ayrı izinli doğrulama olmalıdır.

**Efor:** 1–3 gün.

## Birleştirme ve uygulama sırası

08-01 → 08-02 → 08-03 → 08-04 → 08-05. 08-03 ve 08-05 aynı asenkron process state-machine çalışması içinde çözülebilir. 08-04 küçük bir erken düzeltme olarak araya alınabilir.

Rol 01'in demo/production promotion ve tek sefer etkin production düğmesi bulguları burada tekrarlanmadı. Rol 04'ün NaN/JSON ve model kazananı bulguları, rol 06'nın export üzerine yazma/backend sözleşmesi kusurları, rol 07'nin çift-model bellek geçişi ve stale provider etiketleri farklı bulgulardır. 08-02 aynı load bölümünü inceler fakat bellekten bağımsız istek/UI durum kusurudur. 08-01 export artefaktından ayrı managed model-asset promotion yoludur. 08-04 sabit PNG ismi etkisi, rol 06 export üzerine yazma sorunundan ayrıdır. Ana rapor UI duyarlılık önerisini performans ölçümü olarak göstermemelidir.

Olumlu kontroller: malformed command izolasyonu ve request ID'nin worker hata olayına taşınması; önceki backend'in yeni candidate başarısızlığında korunması; asset-lock'un geçici dosyadan replace ile yazılması. Bu incelemede hiçbir çözüm uygulanmadı.
