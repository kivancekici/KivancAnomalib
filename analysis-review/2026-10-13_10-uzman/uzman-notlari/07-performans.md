# Uzman 07 — CPU/GPU, bellek ve gecikme performansı

13 Ekim 2026, Europe/Istanbul. Bu rol ayrı bir yapay zekâ uzman incelemesidir. Uygulama kodu değiştirilmedi; eğitim, gerçek model yükleme/inference/export, kamera açma, native GUI, paket yükleme ve ağ indirme yapılmadı.

## Karar özeti

Gecikme karşılaştırmasına başlamadan ölçüm kapsamı ve çalıştırılan backend bilgisi düzeltilmeli. Kod iki doğrulanmış ölçüm/teşhis kusuru içeriyor: backendler farklı işlem sınırlarını aynı `model_latency_ms` adıyla bildiriyor; ONNX Runtime provider fallback'inden sonra önceden kaydedilen GPU etiketi güncellenmiyor. Büyük modellerde geçişin çift bellek gereksinimi ve PatchCore kapasite planı ayrıca ele alınmalı. Mevcut kanıtla belirli bir modelin gerçek inference gecikmesinin yüksek olduğu veya hangi cihazın daha hızlı olduğu söylenemez.

## Kapsam ve kanıt sınırları

İncelenen kapsam: `worker/anomalib_worker.py:2239` gerçek eğitim; `2533` export; `2707–3315` inference backendleri ve kalıcı oturum; `mainwindow.cpp:1555`, `2989`, `3511`, `3845`, `3932` görüntü/latency/UI; `anomalibprocess.cpp:26`, `105`, `143` süreç başlangıç/durdurma. Kamera hattının ölçüm dışında kaldığını anlamak için `cameraservice.cpp:65` de okundu.

Kurulu sürümler ana envanterde doğrulandı: Anomalib 2.6.0, Torch 2.13.0+cu130, onnxruntime-gpu 1.28.0, OpenVINO 2026.2.1. Kütüphane davranışları hem kurulu kaynakla hem aşağıdaki resmi kaynaklarla kontrol edildi. Ortam keşfinde GPU/provider görülmesi gerçek modelin GPU üzerinde başarıyla yürüdüğünün kanıtı değildir.

Gerçekten çalıştırılan ek kontrol: `.venv/Scripts/python.exe -B -X utf8 analysis-review/2026-10-13_10-uzman/kanitlar/07-performans-probe.py`, exit 0; araç duvar süresi 1,245 saniye. `PYTHONDONTWRITEBYTECODE=1`, UTF-8 ve inceleme altındaki TEMP/TMP kullanıldı. Bu süre yalnız küçük probe süresidir. Probe kontrollü sanal saat, sahte engine/ORT session ve küçük CPU NumPy/PIL nesneleri kullanır; model latency/GPU benchmark değildir. Kod ve sonuç: [probe](../kanitlar/07-performans-probe.py), [JSON kanıt](../kanitlar/07-performans-probe.json).

Ana koordinenin mevcut suite koşusu 37 test OK / 67,930 saniyedir; bu rol suite'i yeniden çalıştırmadı. Üretim veri kümeleri erişilebilir değildi; tek 607.008.673 bayt ONNX dosyası envanterde mevcut, checkpoint yok. Bu rol ONNX ikilisini yüklemedi.

## Bulgular

### 07-01 — Backendler aynı gecikme adı altında farklı işlemleri ölçüyor

**Tür / önem:** Doğrulanmış hata — ölçüm sözleşmesi, P2.

**Kanıt:** Checkpoint timer `worker/anomalib_worker.py:2779–2789` içindeki `Engine.predict(data_path=...)` çağrısının tamamını kapsar. Kurulu Anomalib `engine/engine.py:655–689` bu çağrı içinde workspace/trainer hazırlığı, `PredictDataset`, DataLoader ve trainer prediction oluşturur. ONNX timer ise PIL okuma/resize sonrasında başlar (`2957–2963`); OpenVINO ONNX (`3073–3079`) ve OpenVINO (`3163–3169`) da okuma/resize işlemlerini dışarıda bırakır. Hepsi `3281` satırında aynı `model_latency_ms` alanına konur. ONNX runtime sigma blur'ı ayrıca bu model timer'ının içindedir (`2967–2972`).

**Bağımsız kontrol:** Aynı varsayılan 15 sanal ms okuma + 5 sanal ms forward işi mock ile checkpointte 20, ONNX'te 5 olarak ölçüldü. Bunlar fiziksel hız ölçümleri değil, timer sınırını doğrulayan kontrollü değerlerdir.

**Tetikleyici:** Aynı artefaktın checkpoint ve ONNX/IR yollarını kıyaslayıp model etiketli süreyi yalnız ağ yürütmesi sanmak. Qt'deki `total` de gerçek operatör süresi değildir: worker timer `3260–3270` sadece backend + artefakt yazmayı kapsar. Qt'nin okunabilirlik kontrolü (`mainwindow.cpp:3518`), önizleme (`3542`), kamera yakalama (`3612`), IPC ve sonucu ekrana çizme dışarıdadır.

**Etki:** Backend hız sıralaması ve kamera çevrim bütçesi yanlış yorumlanabilir; gerçek modelin yavaşladığı kanıtlanmış değildir.

**Çözüm:** `decode_resize_ms`, `prepare_transfer_ms`, `forward_ms`, `postprocess_ms`, `artifact_write_ms`, `worker_total_ms` kapsamlarını tutarlı tanımlamak; Qt'de isteğin başlangıcından sonucun gösterimine `operator_total_ms` eklemek. Mevcut ölçümleri kapsamı açıklayan adlarla sunmak da ilk küçük düzeltmedir. Torch CUDA timer öncesi/sonrası synchronize zaten var (`2777–2788`, `2872–2878`); bu doğru koruma korunmalı. [Torch 2.13.0 synchronize kaynağı](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/__init__.py).

**Kabul testi:** Kontrollü saat ile aynı 15+5 işte backendler ortak kapsam için aynı değeri verir; okuma/artifact/IPC gecikmesini ayrı enjekte eden mock testler doğru alanı değiştirir. CUDA ölçümü kernel tamamlanmasını bekler; soğuk ilk çağrı ayrı kaydedilir.

**Efor:** 1–2 geliştirici günü; ilk yeniden adlandırma/açıklama 0,5 gün.

### 07-02 — ONNX provider fallback'inden sonra GPU etiketi eski kalıyor

**Tür / önem:** Doğrulanmış hata; provider hatasıyla tetiklenir, P2.

**Kanıt:** Worker `session.get_providers()[0]` ile `device/device_name/backend_name` bilgilerini yalnız constructor'da `2917–2922` satırlarında kaydeder. Sonraki warm-up `2939` ve normal run `2949` sonrasında güncelleme yoktur. Kurulu ORT `onnxruntime_inference_collection.py:327–334`, `EPFail` durumunda `set_providers(...)` yaparak yürütmeyi yeniden dener. Aynı davranış [resmi ORT 1.28.0 kaynağında](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.28.0/onnxruntime/python/onnxruntime_inference_collection.py) doğrulandı. Inference sonucu backend'in bu eski alanlarını kullanır (`3277–3279`).

**Bağımsız kontrol:** Mock session `run()` sonrasında yalnız `CPUExecutionProvider` bildirirken worker hâlâ `device=cuda:0`, backend `ONNX Runtime · CUDAExecutionProvider` bildiriyor. Gerçek GPU/provider hatası oluşturulmadı.

**Tetikleyici:** CUDA provider kurulup başlangıçta seçildikten sonra warm-up veya gerçek predict esnasında ORT'nin EPFail nedeniyle CPU'ya geçmesi.

**Etki:** Operatör CPU fallback'i GPU gecikmesi olarak değerlendirebilir; cihaz teşhisi ve performans karşılaştırması yanlış olur. Ayrıca ilk provider'ın CUDA olması tüm node'ların CUDA üzerinde yürüdüğünü göstermez: ORT desteklenmeyen node'ları sonraki CPU provider'a atayabilir. [Resmi Python API açıklaması](https://onnxruntime.ai/docs/api/python/api_summary.html).

**Çözüm:** Warm-up ve her run sonrasında registered provider listesini yenilemek, değişimi açık bir warning/fallback olayı olarak bildirmek. Alanı `preferred_provider` olarak tanımlamak; gerçek graph cihaz dağılımı için isteğe bağlı node assignment/profiling özeti sunmak. Kurulu 1.28.0 Python kaynak `252–258` graph assignment kaydı API'sini gösteriyor. Provider listesiyle gerçek execution kanıtını birbirinden ayırmak.

**Kabul testi:** Warm-up ve normal predict içinde CUDA→CPU değiştiren sahte session ile model_loaded/inference olaylarının CPU'yu ve fallback nedenini göstermesi. Ayrı graph-assignment fixture'da karma CPU/CUDA dağılımı GPU-only diye sunulmamalı.

**Efor:** 0,5–1 gün; isteğe bağlı ayrıntılı profiler 1 gün ek.

### 07-03 — Model değişimi eski ve yeni modeli aynı anda bellekte tutuyor

**Tür / önem:** Koşullu risk, P1 kapasite sınırındaki model/cihaz için; ölçülmüş OOM değildir.

**Kanıt:** `PersistentInferenceSession.load`, yeni `candidate` oluşturma/compile/warm-up boyunca eski `self.backend`'i korur (`3193–3217`); eski referans ancak `3219–3222` sonrasında bırakılır, cache temizliği `3223`'te yapılır. Probe weakref kontrolü eski backend'in candidate constructor'da canlı, cleanup'ta bırakılmış olduğunu doğruladı. Önceki modeli koruma tercihinin yükleme başarısızlığında geri dönüş faydası vardır; bedeli geçici çift model kapasitesidir.

**Tetikleyici:** Belleğin tek model + çalışma alanına yettiği, fakat eski model + yeni model + yeni warm-up çalışma alanına yetmediği RAM/VRAM sınırında başka büyük model yüklemek. Envanterdeki yaklaşık 607 MB disk artefaktı riskin anlamlı olduğunu gösterir; disk boyutu gerçek VRAM/RAM kullanımıyla eşit kabul edilmez.

**Etki:** Tek başına çalışabilecek bir model geçiş sırasında yüklenemeyebilir; swap/çöp toplama artışı veya OOM koşulludur ve burada ölçülmedi.

**Çözüm:** İki açık mod: kapasite uygunsa mevcut atomik geçişi korumak; kapasite sınırlıysa eski modeli kontrollü unload edip yeni modeli yüklemek. Geçişte ölçülmüş peak RAM/VRAM ve beklenen ek workspace bilgisini raporlamak; hatalı yüklemeden sonra yeniden yükleme/geri dönüş akışını tasarlamak. `torch.cuda.empty_cache()` canlı modele ait tensor'ları boşaltmaz ve PyTorch cache'i ORT/OpenVINO allocator yönetiminin yerine geçmez. [Torch 2.13.0 empty_cache resmi kaynağı](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/memory.py).

**Kabul testi:** Kapasite sınırlı mock allocator ile eski+yeninin aynı anda sığmadığı koşulda seçilen unload-first modunun başarılı olması; yükleme başarısızlığında UI'nin doğru model durumunu göstermesi. İleri doğrulamada gerçek iki küçük artefaktla peak bellek ölçülür; büyük mevcut model zorlanmaz.

**Efor:** 1–2 gün.

### 07-04 — PatchCore için eğitim öncesi kapasite tahmini yok

**Tür / önem:** Koşullu risk, P2; OOM veya belirli yavaşlık doğrulanmadı.

**Kanıt:** UI `mainwindow.cpp:861–866` iki boyutta 2048'e, `946` coreset oranında 1,0'a izin verir; worker yalnız parametre aralığını denetler (`199–219`, `307–313`). Eğitim `2267–2276` kapasite bütçesi olmadan başlar. Kurulu Anomalib 2.6.0 PatchCore kaynak `torch_model.py:176`, `280–284` tüm görüntü embedding'lerini biriktirir, birleştirir ve sonra coreset seçer. Koreset oranını azaltmak, bu ilk tam embedding koleksiyonunu ortadan kaldırmaz. Nearest-neighbor araması 1024 query patch'lik chunk'ları tüm memory bank ile kıyaslar (`52`, `346–367`). [Anomalib 2.6.0 resmi PatchCore kaynağı](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/patchcore/torch_model.py).

**Tetikleyici:** Görüntü sayısı, giriş boyutu, yoğun patch stride, feature katmanları veya coreset oranı artırılırken bunların birlikte bellek üzerindeki etkisinin görülmemesi.

**Etki:** Pahalı embedding çıkarımı/coreset aşamasında geç kapasite hatası; inference bank'i ve uzaklık hesaplarının büyümesi. Sabit backbone/stride altında her iki boyutu 256'dan 2048'e çıkarmak yaklaşık 64 kat patch alanı demektir; bu gerçek gecikme için 64 kat ölçümü değildir.

**Çözüm:** Eğitim planında `N` görüntü, `P` görüntü başına patch, `D` feature boyutu, `b` byte/dtype için yaklaşık tam embedding `N*P*D*b`, coreset bank `ratio*N*P*D*b`, inference uzaklık chunk'ı `min(P,1024)*bank_rows*b` göstermek. Framework geçici kopyaları/aktivasyonlar nedeniyle bunun alt sınır tahmini olduğu açıklanmalı. Örnek yalnız aritmetiktir: N=1000, P=4096, D=1536, float16 ile ham embedding yaklaşık 11,72 GiB; %10 bank yaklaşık 1,17 GiB. Projedeki gerçek dataset boyutu/embedding shape'i olmadığı için bu mevcut tüketim iddiası değildir. Patch stride ve coreset değişikliği başarımı etkileyebilir; doğruluk/kaçırma testiyle birlikte değerlendirilmelidir.

**Kabul testi:** Saf shape/arithmetic fixtures ile boyut, görüntü sayısı, dtype ve coreset oranı arttığında tahminin doğru değişmesi; büyük görüntü sayısında uyarının eğitim başlamadan gelmesi. Küçük onaylı veriyle peak allocated/reserved/RSS ölçümünün tahminden neden büyük olduğu kayıt altına alınır.

**Efor:** 1–2 gün; çalışma zamanı peak bellek alanları 0,5 gün ek.

### 07-05 — Gerçek eğitim sonuçları model seçimi için inference benchmark içermiyor

**Tür / önem:** İyileştirme fırsatı, P2.

**Kanıt:** Gerçek eğitim her model için `latency_ms=None` bildiriyor (`2309`), buna karşın sonuç tablosu latency kıyas sütunu/açıklaması sunuyor (`mainwindow.cpp:617–640`, `2989`). Probe mevcut 35 `demo=false` export iş JSON'unda metrics.latency_ms alanının tamamının null olduğunu gördü. `metrics.duration_seconds` 34,89–152.155,44 aralığında geçmiş kayıtlara kopyalanmış; worker `2266–2281` sınırı constructor+fit+test toplamıdır. Tekrarlayan export iş JSON'ları bağımsız eğitim örnekleri değildir. Örnek kaynak: `build/Desktop_Qt_6_11_1_MSVC2022_64bit_Release/release/.data/jobs/export-20260810-070829-685.json:9,23`.

İki production manifestindeki 38,2 ms, `demo/patchcore/model.ckpt` ile kayıtlıdır ve worker'ın demo sabitiyle eşleşir (`1569`). Bu sayı gerçek model latency kanıtı sayılamaz.

**Tetikleyici:** Kullanıcı gerçek modelin accuracy/size yanında latency ile seçilmesini veya geçmiş eğitimin hangi aşamasının uzun sürdüğünün açıklanmasını bekler.

**Etki:** Mevcut gerçek model kıyası inference hızını kapsamaz; geçmiş toplam eğitim/test süresinden export darboğazı çıkarılamaz. Optimize edilmesi gereken aşama belirlenemez.

**Çözüm:** Önce 07-01 ölçüm sözleşmesini kurmak; sonra isteğe bağlı küçük ve aynı örnek listesiyle eğitim sonrası benchmark, `cold`, `warmup_count`, n, p50/p95, backend/device/input_size/dtype/fingerprint bilgilerini saklamak. `create`, `fit`, `test`, `preview`, `export`, `compile/load` sürelerini ayrı raporlamak. Demo sonuçlarını sentetik olarak kalıcı işaretlemek; ölçülmeyen latency için “ölçülmedi” göstermek.

**Kabul testi:** Mock gerçek eğitim sonucunda latency yokken açık ölçülmedi etiketi; benchmark çağrısında aynı örnek/ölçüm kapsamı ve warm-up sayısıyla sonuç kaydı; sürelerin aşama sınırlarına göre ayrılması. Gerçek forward benchmark ayrı izinli doğrulama işi olarak yürütülür.

**Efor:** 1–2 gün.

### 07-06 — OpenVINO cihaz seçme politikası artefakt yoluna göre değişiyor

**Tür / önem:** İyileştirme fırsatı, P3; seçilen cihazın yavaş olduğu ölçülmedi.

**Kanıt:** Override-capable ONNX fallback `2998–3025` GPU adayları ardından CPU'yu dener; ilk başarılı compile'da durur ve latency kıyaslamaz. IR/diğer ONNX OpenVINO yolu ise `3115–3149` CPU ve uygun Intel GPU adaylarında üç predict yapıp ilkini atarak kalan iki sürenin üst değerine göre seçim yapar. Daha önce seçilen cihaz varsa o cihazdan sonra durur (`3138`). Cache anahtarı yalnız model fingerprint'i, cache oturum belleğindedir; donanım/yük/güç/driver değişimi veya sigma ayarı ölçüm profilinde yoktur.

**Tetikleyici:** Aynı iş istasyonunda farklı export/fallback yoluyla yükleme, birden fazla derlenebilir cihaz veya değişen güç/yük koşulları.

**Etki:** Kullanıcı cihaz seçimindeki farklılığın nedenini göremez; “ilk uyumlu” ve “iki örnekle daha düşük süre” politikaları farklı sonuç verebilir. Hiçbir cihaz için gerçek yavaşlık iddiası yapılmıyor.

**Çözüm:** Tek bir açık politika: uyumluluğu doğrulanmış cihaz seçimi, operatörün belirlediği cihaz veya isteğe bağlı sabit örnekler/warm-up ile kısa kalibrasyon. Seçim nedenini ve ölçüm örnek sayısını bildirmek; cache'i model+sürüm+cihaz profiliyle anahtarlamak ve yeniden kalibrasyon eylemi sunmak. Single-image synchronous akış için latency hedefini açık tutmak; throughput ayarlarını seri akışta otomatik “hızlandırma” saymamak. [OpenVINO 2026 resmi performans ipuçları](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html).

**Kabul testi:** Sahte derlenebilir cihaz listesi ve kontrollü süreler ile iki artefakt yolunun aynı politika altında aynı seçimi yapması; ölçüm yapılmayan seçimde “benchmark kazananı” denmemesi; cache/politika değişince seçim nedeninin yenilenmesi.

**Efor:** 0,5–1,5 gün.

## Önerilen ölçüm planı — bu incelemede çalıştırılmadı

1. Önce ölçüm alanlarını ve backend/fallback kayıtlarını düzeltmek; tüm fixture sonuçlarını doğrulamak.
2. Sonraki izinli benchmark için tek küçük sabit yerel görüntü listesi, aynı artefakt fingerprint'i, aynı input/dtype/sigma, cihaz/güç/driver/runtime kaydı kullanmak. Soğuk yükleme/compile ve ilk prediction ayrı tutulmalı. Warm-up başarısızlığının sessiz geçildiği Torch yolları (`2764–2767`, `2864–2865`) “warmed” kabul edilmemeli.
3. Sabit en az 5 warm-up ve 30 ardışık ölçümle p50/p95/min/max ve n kaydetmek; sayı üretim bütçesine göre ayarlanır. CUDA async işin bitmesi timer sonuna dahil edilmeli; mevcut synchronize korunmalı. CPU/ORT/OpenVINO gerçek yürütme yeri node assignment/profiling ile doğrulanmalı. [Resmi ORT profiler açıklaması](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
4. Decode/resize, device transfer, forward, sigma/postprocess, PNG yazma, IPC/Qt çizimi ve kamera yakalama sürelerini ayırmak. Kamera kapalıyken dosya toplamı, izinli kamera oturumunda operatör toplamı farklı kayıtlar olmalı.
5. Eğitimde embedding sayısı/bank shape/dtype, peak RSS, CUDA allocated/reserved/peak, her aşama süresi; yükleme/değişim/unload döngüsünde peak bellek kaydetmek. ORT/OpenVINO tüketimini yalnız Torch allocator sayacıyla temsil etmemek.
6. Girdi çözünürlüğü/coreset/stride varyantlarını yalnız kontrollü tekrar eğitim ve ayrılmış değerlendirme verisiyle ele almak. Daha hızlı varyant için false-positive/false-negative ve pixel metriği karşılaştırılmadan öneri kesinleştirilmemeli.

Önerilen uygulama sırası: 07-01 ve 07-02 teşhis/ölçüm doğruluğu; 07-03/07-04 kapasite koruması; 07-05 kalıcı benchmark; 07-06 cihaz politikası. Bu incelemede çözüm uygulanmadı.
