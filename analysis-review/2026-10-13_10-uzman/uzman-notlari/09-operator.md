# Uzman 09 — UI/CLI, operatör akışı, yapılandırma ve teşhis

İnceleme tarihi: 13 Ekim 2026. Kapsam: `mainwindow.cpp`, `mainwindow.h`, `README.md`; UI ile worker sözleşmesini doğrulamak için `worker/anomalib_worker.py` ve `anomalibprocess.cpp`. Bu not kaynak/ayar değişikliği içermez. Qt uygulaması, kamera, eğitim, model yükleme ve üretim inference çalıştırılmadı. Ekran görüntüsü veya önceki QA raporu güncel çalışma kanıtı olarak kullanılmadı.

## Doğrulama yöntemi ve sınır

UI olay bağlantıları, durum alanları, job hazırlama ve sonuç/provenance akışları kaynak üzerinden izlendi. Beş bulgunun UI kısmı **statik program akışı kanıtıdır**; render veya gerçek tıklama deneyi değildir. Önem: P1 = model/sonuç kökenini yanlış kaydedebilen yüksek öncelik; P2 = operatör iş akışını bozabilen orta öncelik. Eforlar bir geliştirici için uygulama ve hedefli doğrulama tahminidir.

Gerçekten çalıştırılan tek rol-09 deneyi:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& .venv/Scripts/python.exe -B -X utf8 analysis-review/2026-10-13_10-uzman/kanitlar/operator-demo-check-probe.py
```

Çıkış kodu **0**. Probe, `run_demo({"mode":"check","demo":true})` sırasında `anomalib` ve `torch` importlarını engelleyen bir kontrol koydu; bu importlara **sıfır girişimle** `environment`, `ok=true`, `demo=true` üretildi. Bu mevcut ortamın bozuk olduğunu göstermez; demo kontrolünün gerçek ortamı denetlemediğini gösterir. Kanıt: [probe kaynak](../kanitlar/operator-demo-check-probe.py), [probe çıktısı](../kanitlar/operator-demo-check-probe.json). Eğitim/model/kamera açılmadı, bytecode yazımı kapatıldı.

## 09-01 — Dataset değişimi önceki sonuçlarla yanlış üretim kökeni oluşturuyor

**Tür / önem:** Doğrulanmış hata (statik) / **P1**.

**Kanıt:** `mainwindow.cpp:2425–2437` ve `2448–2451`, seçili dataset kökünü değiştirip sayım ve örnek görünümünü güncelliyor; `m_results` veya `m_winnerModel` için yeni dataset ile bağ kurmuyor/invalidasyon yapmıyor. Sonuçları temizleyen yer yeni eğitim başlangıcı (`2646–2647`). `promoteWinner()` checkpoint'i önceki sonuçtan alıyor (`3064–3065`, `3109–3112`), manifestin `dataset_root` alanını ise o andaki `m_datasetRoot` ile yazıyor (`3078`). Gerçek worker sonucu `worker/anomalib_worker.py:2298–2319`, checkpoint/metrics/parametreleri içeriyor ancak kaynak dataset kökünü aynı sonuç içine koymuyor. Dataset örnek önizlemi değişim sırasında temizleniyor (`mainwindow.cpp:1604–1634`); bulgu, heatmap temizlenmediği iddiası değildir.

**Tetikleyici:** A datasetinde gerçek eğitim tamamla; A sonucu hâlâ tablo/kazanan olarak dururken B datasetini seç veya B için Rescan yap; daha önce promote edilmemiş kazananı promote et. İşlem sonucu: A checkpoint ve A metrics, `dataset_root=B` ile aynı manifestte bulunur. Eğitim sırasında dataset değiştirildiğinde de job dosyası A snapshot'ı, görünür kök B olabilir.

**Etki:** Manifestin model kökeni yanlış olur. Ürün/dataset karşılaştırmaları ve tekrar üretme işlemi hatalı kayda dayanır; başka ürünün modelini kullanma veya kalibrasyon seçme gibi operatör kararlarının etkisi koşulludur. Koddan ölçülemeyen model başarımı iddiası yoktur.

**Çözüm:** Eğitim job'ına run kimliği ve dataset kökü/fingerprint snapshot'ı ekleyip her sonucu bu snapshot ile sakla; promotion verisini seçili sonuçtan üret. Seçili dataset değiştiğinde önceki sonuçları geçmiş koşu olarak açıkça etiketle veya aktif kazanan/promotion geçerliliğini kaldır. Dataset seçiminden bağımsız bir sonuç geçmişi korunabilir.

**Kabul testi:** İki küçük fixture dataset ile A koşu sonucunu enjekte et; B seçimi sonrası promotion ya açık gerekçeyle engellensin ya da manifestte checkpoint ve metrics ile birlikte kaynak kök A kalsın. B'ye ait kökün A checkpoint ile kaydedilmediği doğrulansın. Çalışan job sırasında A→B seçiminde gelen sonuçların run/dataset bağının A kaldığı da doğrulansın.

**Efor:** 1–2 gün. Rol-01'in geçersiz/demo checkpoint promotion bulgusu ile ortak promotion alanına dokunur ancak bu hata geçerli gerçek checkpoint ile de oluşur; ana raporda aynı köken/manifest iş paketi altında birleştirilebilir.

## 09-02 — Belgelenen ortam doğrulama sırası gerçek ortamı denetlemiyor

**Tür / önem:** Doğrulanmış hata (doküman/UI sözleşmesi; worker probe ile destekli) / **P2**.

**Kanıt:** Demo varsayılanı true (`mainwindow.cpp:2021–2025`). README `69`, önce **Verify environment**, sonra demonstration kutusunu kapatma talimatı veriyor. `verifyEnvironment()` check job'ına mevcut demo değerini gönderiyor (`3679–3683`). Worker `main()` önce demo dalına giriyor (`worker/anomalib_worker.py:3453–3463`); `run_demo()` check için `ok=true/demo=true` üretip geri dönüyor (`1603–1612`). Gerçek sürüm/CUDA/provider kontrolleri `run_check()` içinde (`3368–3425`) ve bu yol çalışmıyor. Qt environment olayı `demo` alanını ayırmadan yalnız `ok` ile başarı stilini seçiyor (`mainwindow.cpp:2789–2798`). Mesaj metninde “Demo worker ready” ifadesi var; doğrudan “gerçek ortam hazır” dediği iddia edilmiyor.

**Tetikleyici:** Demo varsayılanında README'nin sırasını izle. Anomalib kurulu olmayan veya uyumsuz Python bile demo check'i geçebilir; bundan sonra kutuyu kapatmak gerçek check'i otomatik başlatmıyor.

**Etki:** Belgede vaat edilen gerçek sürüm/cihaz doğrulaması atlanır ve uyumsuz ortam gerçek iş başlatılana kadar fark edilmeyebilir. Bu incelemenin ana koordinatörü gerçek ortam kontrolünü ayrıca başarılı çalıştırmıştır; bulgu mevcut ortamın başarısız olduğu anlamına gelmez.

**Çözüm:** “Verify environment” gerçek kontrolü her zaman `demo=false` ile çalıştırsın; demo worker doğrulaması gerekiyorsa ayrı eylem/durum olsun. README sırasını buna göre düzelt. Ortam/Python/mod değişiminde önceki doğrulama durumunu “yeniden kontrol gerekli” olarak işaretle.

**Kabul testi:** Demo kutusu açıkken Verify gerçek sürüm kontrolüne ulaşmalı; Anomalib'i olmayan kontrollü Python fixture'ında gerçek ortam hatası göstermeli. Demo kontrolü ayrı tutuluyorsa durum açıkça yalnız demo worker doğrulaması olarak etiketlenmeli. Sürüm bilgisi ve kullanılan Python yolu test edilen durumla eşleşmeli.

**Efor:** 0,5–1 gün. Rol-10 kurulum/tekrar üretilebilirlik ile paylaşılabilir; burada özel hata README/UI iş sırasıdır.

## 09-03 — Demo/gerçek mod değişimi aktif inference durumunu taşımaya devam ediyor

**Tür / önem:** Doğrulanmış hata (statik durum geçişi) / **P2**.

**Kanıt:** Demo `toggled` bağlantısı yalnız QSettings değerini kaydediyor (`mainwindow.cpp:2027–2030`); hazır model, aktif yol/backend veya session değişmiyor. Demo kazananı seçimi gerçek runtime yüklemeden `m_inferenceModelReady=true` yapıyor (`3256–3283`). Inspect'e giriş yalnız ready=false ve worker çalışmıyorsa restore ediyor (`2409–2411`, `3294–3302`). `runInference()` mevcut kutu değerine göre gerçek/demo dalını seçiyor (`3523`, `3558–3592`); gerçek dal sadece session başlatıp `predict` gönderiyor, model yüklemiyor (`3578–3592`). Worker backend boşsa açık hata veriyor (`worker/anomalib_worker.py:3241–3243`). Qt predict-error yolu ise request önekine dayanarak modeli tekrar available sayıyor (`mainwindow.cpp:3448–3455`).

**Tetikleyici:** Demo açıkken Inspect'e gir veya “Use trained winner” yap; böylece hazır demo durumuna geç. Settings'te demo kutusunu kapat; Inspect'te okunabilir bir görüntü test et. Gerçek session model yükleme komutu almadan predict isteği alır. Tersi yönde gerçek model yüklüyken demo açılırsa aynı aktif model/backend ayrıntıları korunarak sentetik sonuç üretilebilir.

**Etki:** Demo→gerçek geçişinde test hatası oluşur; hata sonrasında backend boşken tekrar hazır gibi işaretlenebilir. Gerçek→demo geçişinde ekranda tutulan gerçek backend/model ayrıntıları ile kullanılan sentetik işlem birbirinden ayrışır. Bu ikinci durum yanlış yorumlama riskidir; model başarımı ölçümü değildir.

**Çözüm:** Mod geçişini açık bir durum geçişi olarak ele al: pending işi yönet, gerçek session/modeli yükle veya kaldır, ready bayrağını aktif runtime'dan türet, önceki sonuçları yeni mod için geçersiz say. Sonuçta kullanılan işlem türü görünür/saklanan metadata olsun. Predict hatasında modelin varlığını yalnız request önekinden çıkarma; worker yanıtı gerçek model durumunu bildirsin.

**Kabul testi:** Fixture worker ile demo-ready→real→predict senaryosunda `load_model` onayı olmadan test düğmesi etkin olmasın/predict gönderilmesin. Real-ready→demo geçişinde model ayrıntısı ve sonuç açıkça sentetik olsun; yeniden real'e dönünce doğru gerçek artifact yeniden yüklenmeden ready olmasın. Backend yok hatasında ready=false kalmalı.

**Efor:** 1–2 gün. Rol-08 request routing bulgusu ile son error/readiness kısmı örtüşebilir; ana iş paketi mod/runtime durum makinesidir.

## 09-04 — Seçili girdi/model değişiminde veya erken reddetmede eski sonuç paneli geçerli gibi kalabiliyor

**Tür / önem:** Doğrulanmış hata (statik UI durum invalidasyonu) / **P2**.

**Kanıt:** Test görüntüsü doğrudan düzenlenebilir QLineEdit (`mainwindow.cpp:1918–1925`); `textChanged`/`editingFinished` ile sonuç invalidasyonu bağlantısı yok. Browse yolu sonuçları temizliyor (`3500–3508`) ancak manuel düzenleme bunu çalıştırmıyor. `runInference()` okunamayan dosya, model hazır değil veya diğer job çalışıyor durumlarında önceki karar/score/maps'i temizlemeden geri dönüyor (`3516–3530`); gerçek yeni predict başlayacaksa önceki sonuçları temizliyor (`3542–3554`). Yeni model yükleme başarı olayı model durumunu ve kısa statüyü güncelliyor fakat önceki score, latency ve heatmap/mask'i temizlemiyor (`3361–3427`). Threshold ve sigma değişimi de geçmiş sonucu kendi kullanılan ayarlarıyla etiketlemiyor (`1868–1893`).

**Tetikleyici:** A/M1 için başarılı NORMAL veya anomaly sonucu al. Test image metnini elle okunamayan B yoluna değiştir ve Test'e bas: footer hata gösterirken büyük önceki karar ve score/maps korunur. Alternatif: M2 modelini yükle; aktif model M2 iken önceki M1 score/maps hâlâ görünür. Burada zamanlama yarışına gerek yoktur.

**Etki:** Operatör görüntü/model/ayar bağını göremediği için eski sonucu mevcut seçimle ilişkilendirebilir. UI'nin gösterdiği değer eski gerçek sonuca aittir; worker'ın B'yi yanlış sınıflandırdığı iddia edilmiyor.

**Çözüm:** Sonucu immutable bir kayıt olarak input path/fingerprint, artifact fingerprint, request/run ID, mode ve kullanılan threshold/sigma ile sakla ve görünümde bu bağı göster. Girdi/model değişiminde paneli temizle veya açık “önceki sonuç” durumuna geçir. Erken validasyon hatası sonucu geçerli mevcut test gibi bırakmamalı.

**Kabul testi:** Fixture A/M1 NORMAL sonucundan sonra manuel B-yolu değişimi/erken dosya hatası, M2 yükleme ve threshold/sigma değişimi için görünür sonuç provenance doğru kalmalı veya panel geçersizleşmeli. Özellikle reddedilmiş B testinin yanında etiketsiz A NORMAL sonucu kalmamalı. GUI otomasyonundan bağımsız Qt state testi ile başlayıp hedefli ekran kontrolü eklenebilir.

**Efor:** 1–2 gün. Rol-08'in geç gelen cevap/request_id eşleşmesi bulgusu burada yeniden sayılmamalı. Asenkron request eşleşmesi ek olarak gereklidir, fakat bu bulgu senkron seçim/değişim ve erken return üzerinden bağımsızdır. Genel “tüm predict hata/stop'larında eski NORMAL kalır” iddiası yapılmıyor; normal predict başlangıcı geçmiş görselleri gerçekten temizliyor.

## 09-05 — Rescan yeni dataset seçse de Capture eski köke yazabiliyor

**Tür / önem:** Doğrulanmış hata (statik hedef-path akışı; fiziksel kare yazımı donanım erişimine bağlı) / **P2**.

**Kanıt:** Browse/chooseDataset her iki edit alanını eşliyor (`mainwindow.cpp:2425–2434`). Ancak `refreshDataset()` yalnız `m_datasetRoot` değiştiriyor (`2440–2451`); Capture edit alanını ve eğitimdeki dataset etiketini eşlemiyor. `nextCapturePath()` öncelikle Capture edit metnini alıp `m_datasetRoot` üzerine tekrar yazıyor (`3911–3919`), buraya sınıf klasörü oluşturuyor (`3921–3929`). Capture eylemleri üretilen path'i doğrudan kameraya veriyor (`3867–3878`, `3888–3899`). UI, job çalışırken Capture kökü/kare eylemlerini kapatmıyor (`setJobRunning`, `2894–2956`; Capture yerel düğmeleri `1737–1746`), ancak bu not çalışan datasetin mutasyon riskini ayrı performans/yaşam döngüsü hatası olarak saymıyor.

**Tetikleyici:** Browse ile A seç; Datasets sayfasındaki edit'e B yazıp Rescan yap (normal sayımları B için görünür). Capture sayfasına geçip örnek yakala; edit eski A kaldığından hedef A olur, `m_datasetRoot` da yeniden A'ya döner. Kamera yoksa gerçek dosya yazımı başarısız olabilir; eski hedefin seçilmesi kaynakta doğrulanmıştır.

**Etki:** Örnekler operatörün en son taradığı datasetten farklı köke eklenebilir; sayımlar ve eğitim dataset etiketi seçimler arasında tutarsız kalabilir. Kullanıcı kaynak/veri dosyalarını koruma açısından hedefin görünür ve tutarlı olması gerekir.

**Çözüm:** Dataset seçimini tek bir setter/state kaynağıyla tüm edit/label/sayım alanlarına uygula. Capture'ın bağımsız hedef seçimi isteniyorsa bunu açıkça ayrı hedef olarak göster ve mevcut eğitim/aktif dataset state'ini yan etkiyle değiştirme. Çalışan job datasetine yazma için açık snapshot/yazma politikası belirle.

**Kabul testi:** Kamera stub'ı ile A Browse→B Rescan→Capture akışında capture hedefi B olmalı veya Capture'ın ayrı A hedefi açıkça gösterilip merkezi dataset B korunmalı. Browse, Rescan ve Capture kök değişimi sonrasında tüm merkezi dataset alanları/etiketleri eşleşmeli. Fiziksel kamera bu kabul testi için gerekli değildir.

**Efor:** 0,5–1 gün. 09-01 ile tek dataset-state düzenlemesi paylaşılabilir; manifest kökeni ve capture hedefi farklı gözlemlenebilir davranışlardır.

## Ek gözlemler — ayrı öncelikli bulgu olarak sayılmadı

- Python edit ayarı değiştirilse de mevcut persistent session `startInferenceSession()` başında yeniden kullanılmaya devam eder (`3164–3166`, `3181–3190`); ortam kontrolü ise yeni edit değerini alır (`2729–2749`). Kullanılan Python/session kimliğinin teşhiste gösterilmesi ve ayar değişiminin yürürlük zamanı açıklaştırılması faydalıdır. Mevcut davranış bilinçli sıcak session tercihi olabilir; ölçüm veya UI deneyi olmadan ayrı hata sayılmadı.
- Python Browse sadece setText yapar (`3669`), kayıt `editingFinished` bağlantısına dayanır (`2005–2014`). Platform/focus olayları bu rol kapsamında denenmedi; “ayar kesin kaybolur” şeklinde bir bulgu yapılmadı.
- Ortam/assets/export eylemlerinin çalışan job'a etkisi, sendCommand dönüş değeri ve request ID eşleştirme rol-08'in bağımsız incelemesiyle birleştirilmelidir. Burada süreç çakışması veya use-after-free varsayılmadı.

## Önerilen uygulama sırası

1. Run/sonuç/provenance snapshot ve dataset state tek kaynak yaklaşımı: 09-01 + 09-05.
2. Demo/gerçek runtime geçişi ve request/result state tutarlılığı: 09-03 + rol-08 korelasyon kontrolü.
3. Sonuç paneli invalidasyonu ve açık provenance gösterimi: 09-04.
4. Gerçek ortam doğrulama eylemi ve README sırası: 09-02.

Bu öneriler uygulanmadı; analiz kapsamındaki kaynaklar değiştirilmedi.
