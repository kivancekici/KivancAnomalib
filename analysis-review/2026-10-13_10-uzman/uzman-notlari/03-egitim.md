# Uzman 03 — Model seçimi ve eğitim doğruluğu

İnceleme tarihi: 13 Ekim 2026. Bu not bağımsız yapay zekâ uzman incelemesidir. Kaynak kod ve konfigürasyon değiştirilmedi; gerçek eğitim, gerçek model oluşturma, ağırlık yükleme, veri indirme, kamera veya GPU iş yükü başlatılmadı.

## Kapsam ve doğrulama

- Uygulama: `worker/anomalib_worker.py` 146–398, 1388–1490, 2041–2333; `mainwindow.cpp` model parametre kontrolleri 816–1554; ilgili `tests/test_worker.py`.
- Yerel paket kaynakları: `.venv/Lib/site-packages/anomalib` altında EfficientAD, PaDiM, PatchCore ve Engine; `.venv/Lib/site-packages/lightning/pytorch` scheduler yapılandırması.
- METADATA ile bağımsız sürüm doğrulaması: Anomalib 2.6.0, Lightning 2.6.5. Ortam envanterinde bildirilen PyTorch 2.13.0+cu130 ve timm 1.0.28 için ek bir sürüm ölçümü yapılmadı.
- Bağımsız hafif probe: [03-egitim-probe.py](../kanitlar/03-egitim-probe.py), [03-egitim-probe.log](../kanitlar/03-egitim-probe.log). `.venv/Scripts/python.exe -B -X utf8` ve `PYTHONDONTWRITEBYTECODE=1` kullanıldı; çıkış kodu 0.
- Probe, worker'ın gerçek normalizasyon işlevlerini çağırdı. Kurulu Anomalib kaynaklarından AST ile ayrılan PaDiM doğrulama dalı, Engine argüman cache'i ve KCenterGreedy'nin erken constructor kontrolü çalıştırıldı. Bunlar tam model oluşturma veya uçtan uca eğitim değildir; boyut ve trainer bilgileri kontrollü stub nesnelerinden geldi.
- EfficientAD scheduler kurulumu, kurulu `configure_optimizers` işlevi üzerinden iki skaler sentetik CPU parametresiyle çalıştırıldı. Yalnız `scheduler.step()` çağrıları yapıldı; forward, backward, `optimizer.step()`, `Engine.fit`, `Engine.test`, export veya inference çalıştırılmadı.
- İki mevcut teacher `.pth` dosyasının ZIP içindeki `data.pkl` bölümü `pickletools.dis` ile yalnız opcode/metaveri olarak okundu. `pickle.load` veya `torch.load` kullanılmadı; tensor ağırlıkları açılmadı/yüklenmedi.
- Ana ajan mevcut worker suite'ini çalışma alanı TEMP/TMP ile çalıştırdı: 37 test OK, 67.930 saniye; [unittest-workspace-temp.log](../kanitlar/unittest-workspace-temp.log). İlk default-TEMP koşusunun sandbox izin hataları uygulama hatası olarak sayılmadı. Bu uzman suite'i yeniden çalıştırmadı.

Resmi sürüm kaynakları için doğru tag `lib/v2.6.0`'dır. `v2.6.0` adresleri 404 verdi; aşağıdaki kaynaklara başarıyla erişildi. Yerel paket satırları ile web kaynağının satırları dokümantasyon/biçim farkları nedeniyle birebir aynı değildir; belirtilen davranışlar iki kaynakta da doğrulandı.

## Bulgular

### 03-01 — EfficientAD teacher kanal ayarı hazır ağırlıklarla uyumsuz değerleri kabul ediyor

**Tür / önem:** Doğrulanmış hata, P2. Hata statik kaynak ve mevcut ağırlık metaverisi ile doğrulandı; gerçek eğitim arızası çalıştırılmadı.

**Kanıt:** `mainwindow.cpp:1111` UI aralığını 1–65536 yapıyor; `mainwindow.cpp:1378` ve `worker/anomalib_worker.py:324` aynı geniş aralığı kabul ediyor. `worker/anomalib_worker.py:1424`–`1447` teacher dosyasını yalnız model boyutuna göre seçip varlık/uzunluk/checksum kontrol ediyor. Kurulu `anomalib/models/image/efficient_ad/lightning_model.py:167`–`173`, seçilen sabit teacher dosyasını katı varsayılan `load_state_dict` ile yüklüyor. Small ağında `torch_model.py:164` son konvolüsyonun çıkış kanal sayısını kullanıcı ayarına bağlıyor; medium için `:223`–`224` aynı ilişki var.

**Bağımsız ölçüm:** Worker `teacher_out_channels=383` değerini kabul etti. Mevcut `pretrained_teacher_small.pth` metaverisi `conv4.weight` boyutunu `[384,256,4,4]`, medium dosyası `conv6.weight` boyutunu `[384,384,1,1]` gösteriyor. 383 ile oluşturulacak katman boyutları bu dosyalarla eşleşemez. Bu dosyaların tensor içeriklerinin geçerliliği ayrıca test edilmedi.

**Tetikleyici:** EfficientAD Small veya Medium seçip Teacher channels değerini 384 dışındaki bir sayıya ayarlamak; gerekli yerel dosyalar hazır olsa bile.

**Etki:** Ön kontrolün kabul ettiği eğitim, teacher yükleme aşamasında şekil uyuşmazlığı nedeniyle tamamlanamaz. Çok büyük kanal değerleri ayrıca henüz ağırlık yüklenmeden gereksiz model belleği ayırabilir; bellek etkisi ölçülmedi.

**Uygulanabilir çözüm:** Uygulamanın yönettiği iki hazır teacher için kanalı 384'e sabitlemek ve UI/worker'da bunu doğrulamak. Özelleştirilmiş teacher gelecekte desteklenecekse teacher dosyası ile kanal/metaveri eşleşmesini ayrı sözleşme olarak eklemek.

**Kabul testi:** 383/385 ayarları eğitim veya model tahsisi başlamadan açıklayıcı hata almalı; 384 Small/Medium ayarları kabul edilmeli. Teacher dosyasının son katman metaverisi için boyut uyum kontrolü yapılmalı.

**Efor:** 0.5–1 gün.

Resmi kaynak, teacher çıkış kanalı parametresini ve sabit boyut adına göre teacher yükleme yolunu doğruluyor: [Anomalib 2.6.0 EfficientAD](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/efficient_ad/lightning_model.py).

### 03-02 — PaDiM retained-features kontrolü seçilen katmanların kanal toplamını dikkate almıyor

**Tür / önem:** Doğrulanmış hata, P2. Gerçek worker kabulü ve kurulu kitaplığın ayrılmış doğrulama dalı çalıştırıldı; gerçek model oluşturulmadı.

**Kanıt:** `mainwindow.cpp:1049` ve `:1356`–`1358`, retained features için 1–65536 veya automatic kabul ediyor. `worker/anomalib_worker.py:310`–`315` aynı aralığı denetliyor. Kurulu `anomalib/models/image/padim/torch_model.py:126`–`137`, üst sınırı seçilen katmanların toplam kanal sayısı yapıyor. Automatic değer, resnet18 için 100; wide_resnet50_2 için 550. resnet18 layer1 yalnız 64 kanal verir; bu kanal boyutu resmi kaynakta da örnekleniyor.

**Bağımsız ölçüm:** Worker `resnet18`, `layers=['layer1']`, `n_features=None`, `pre_trained=False` kombinasyonunu kabul etti. Kurulu aynı doğrulama dalı, kontrollü 64 kanal boyutuyla `ValueError` verdi: 100 retained feature, 64 özgün kanalı aşıyor. Default üç katmanlı PaDiM ayarında bu örnek arıza yoktur.

**Tetikleyici:** PaDiM için resnet18/layer1 seçip automatic bırakmak; ayrıca katman toplamının üstünde açık bir `n_features` değeri seçmek.

**Etki:** UI'da geçerli görünen parametrelerle eğitim model oluşturma aşamasında durur. Kullanıcı arızanın sebebini model oluşturulduktan sonra görür.

**Uygulanabilir çözüm:** Desteklenen backbone/katmanlar için toplam çıkış kanalını hesaplayıp retained feature üst sınırını buna bağlamak. Automatic modun modeldeki gerçek değerini gösterip katman kombinasyonu buna yetmiyorsa açık uyarı vermek veya belgelenmiş `min(default,total_channels)` politikası kullanmak. UI ve worker aynı kuralı paylaşmalı.

**Kabul testi:** resnet18/layer1/automatic ya açıkça reddedilmeli ya da 64'e uyarlanmış etkin değeriyle kaydedilmeli. Açık 64 kabul, 65 ret; varsayılan layer1–3/automatic100 kabul. İki backbone ve katman alt kümeleri kapsanmalı.

**Efor:** 1 gün.

Resmi kaynak, varsayılanları ve seçili kanal toplamı sınırını doğruluyor: [Anomalib 2.6.0 PaDiM](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/padim/torch_model.py).

### 03-03 — Kaydedilen eğitim parametreleri PatchCore/PaDiM için etkin trainer değerleri değil

**Tür / önem:** Doğrulanmış hata, P2. Bu tek-geçişli modellerin bir epoch çalışması doğru kitaplık davranışıdır; hata uygulamanın etkin değer olarak farklı istek değerlerini kaydetmesidir.

**Kanıt:** `worker/anomalib_worker.py:2262` istek parametrelerini normalize ediyor, `:2269`–`2275` Engine'e geçiriyor ve `:2318` aynı sözlüğü `training_parameters` olarak kaydediyor. README bu alanları etkin parametre olarak tanımlıyor. Kurulu PatchCore `lightning_model.py:302`, PaDiM `lightning_model.py:195` `max_epochs=1` ve `devices=1` zorunlu argümanlarını döndürüyor. Engine `engine.py:101`–`106` kullanıcı değerlerini bunlarla değiştiriyor; `:299`–`310` Trainer'ı güncellenmiş cache üzerinden oluşturuyor.

**Bağımsız ölçüm:** Kurulu Engine cache sınıfı `max_epochs=5, devices=2` ile başlatılıp bu model argümanlarıyla güncellendiğinde etkin sözlük `max_epochs=1, devices=1` oldu. Worker'ın result sözlüğü ise güncellenmiş trainer değerlerini okumuyor.

**Tetikleyici:** Varsayılan 5 epoch ile PatchCore/PaDiM eğitimini bitirmek; cihaz sayısını 1 dışındaki bir değere ayarlamak farkı artırır.

**Etki:** Eğitim karşılaştırması ve tekrar üretim kaydı gerçekleşen işi yanlış tarif eder. Kullanıcı epoch veya cihazı artırmanın bu iki modelde etkin olduğunu düşünebilir. Tek epoch kendiliğinden model başarım hatası değildir.

**Uygulanabilir çözüm:** `requested_training_parameters` ve gerçekleşen `effective_training_parameters` alanlarını ayrı kaydetmek; fit'ten sonra trainer/model argümanlarını okuyarak etkin değerleri doğrulamak. UI'da PatchCore/PaDiM'nin tek veri geçişi ve tek cihaz kısıtını göstermek.

**Kabul testi:** 5 epoch/2 device isteği için result istenen 5/2 ve etkin 1/1'i açıkça ayırmalı; EfficientAD'de kısıt uygulanmadığında etkin epoch isteğiyle eşleşmeli. JSON olay/manifest round-trip bu ayrımı korumalı.

**Efor:** 0.5–1 gün.

Resmi kaynaklar, model argümanlarının Engine tarafından öncelikli uygulanmasını ve PatchCore kısıtlarını doğruluyor: [Anomalib 2.6.0 Engine](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/engine/engine.py), [PatchCore](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/patchcore/lightning_model.py).

### 03-04 — Kurulu EfficientAD scheduler'ında adım sayısı ile epoch aralığı uyuşmuyor

**Tür / önem:** Doğrulanmış hata, P2; Anomalib 2.6.0 kaynaklı, uygulamanın kullandığı eğitim yoluna taşınan sürüm davranışı. Model başarım düşüşü ölçülmedi.

**Kanıt:** Uygulama `worker/anomalib_worker.py:2268`–`2276` ile normal EfficientAD/Engine eğitimini kullanıyor. Kurulu EfficientAD `lightning_model.py:343`–`353`, eğitim adımı sayısını `max_epochs * len(train_dataloader)` olarak hesaplıyor; StepLR eşik değeri bunun %95'i. Dönen scheduler sözlüğünde `interval='step'` yok. Kurulu Lightning `core/optimizer.py:299`–`300`, yalın scheduler nesnesini `LRSchedulerConfig`'e çeviriyor; `utilities/types.py:86` varsayılan aralığı epoch yapıyor. İncelenen EfficientAD/AnomalibModule'de bu ilişkiyi değiştiren özel scheduler-step yolu bulunmadı.

**Bağımsız ölçüm:** 5 epoch ve epoch başına 100 batch bilgisiyle kurulu optimizer-konfigürasyon işlevi `step_size=475` üretti. İki skaler sentetik CPU parametresinin scheduler'ına yalnız 5 epoch tick uygulanınca `last_epoch=5`, LR `[0.0001]` kaldı. Eğitim yapılmadı; bu LR planının aritmetik/konfigürasyon doğrulamasıdır.

**Tetikleyici:** Birden fazla batch içeren normal EfficientAD eğitimi. 5×100 örneğinde planlanan geç LR indirimi, 5 epoch süresince gerçekleşmez. 1 toplam eğitim adımında eşik hesabı ayrıca 0 üretir; bu sınır için ayrıca gerçek scheduler-step arızası çalıştırılmadı.

**Etki:** Öğrenme oranı planı kaynak yorumundaki eğitim sonuna yakın azaltma davranışıyla eşleşmez; konfigürasyonun optimizer adımı yerine epoch sayması eğitim dinamiğini değiştirir. Yanlış alarm veya kaçırma oranına etkisi gerçek veri olmadan sayısallaştırılamaz.

**Uygulanabilir çözüm:** Bu davranışı açıkça doğrulayan sürüm sözleşme testi eklemek; doğrulanmış düzeltmeli Anomalib sürümünü pinlemek veya uygulama içinde kontrollü EfficientAD uyarlayıcısıyla scheduler'a `interval='step'` vermek. Eşik `max(1,int(0.95*num_steps))` olarak küçük veri sınırını korumalı. Site-packages dosyası elle değiştirilmemeli.

**Kabul testi:** Gerçek veri/GPU gerektirmeyen optimizer-scheduler sözleşme testinde 500 adımlık plan için LR 474 optimizer tick boyunca ilk değerde kalıp 475. tickte 0.1 çarpanına inmeli; scheduler aralığı step olarak doğrulanmalı. 1 adım sınırı sıfıra bölme oluşturmamalı. Daha sonra kullanıcının ayrıca yetkilendireceği eğitimle başarım karşılaştırması yapılabilir.

**Efor:** 1–2 gün; desteklenen sürüm/uyarlayıcı kararına bağlı.

Resmi sürüm kaynakları aynı LR eşiğini ve Lightning'in epoch varsayılanını doğruluyor: [Anomalib 2.6.0 EfficientAD optimizer](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/image/efficient_ad/lightning_model.py), [Lightning 2.6.5 scheduler config](https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/2.6.5/src/lightning/pytorch/utilities/types.py).

### 03-05 — Küçük görüntü/veri ile düşük PatchCore coreset oranı eğitim sonunda reddedilebilir

**Tür / önem:** Koşullu risk, P2. Mevcut kullanıcı veri kümesinde tetiklendiği gösterilmedi; kurulu constructor'ın koşulu bağımsız çalıştırıldı.

**Kanıt:** `mainwindow.cpp:946` en düşük UI oranını 0.001 yapıyor; `worker/anomalib_worker.py:283`–`289` daha küçük tüm pozitif oranları da kabul ediyor. Girdi boyutu en düşük 32 piksel olabiliyor (`worker/anomalib_worker.py:208`). Kurulu PatchCore `torch_model.py:280`–`284` embedding'leri topladıktan sonra KCenterGreedy oluşturuyor. `sampling/k_center_greedy.py:49`–`56`, `int(embedding_count * sampling_ratio)` sıfırsa `ValueError` veriyor.

**Bağımsız ölçüm:** Worker 0.001 oranını kabul etti. Kurulu erken constructor kontrolü 128 embedding boyutu ile çalıştırıldığında coreset_size 0 nedeniyle hata verdi. Bu 128 boyutu kontrollü stub'dır; mevcut veri kümesinden ölçülmedi.

**Tetikleyici:** Küçük veri kümesi, küçük girdi/seyrek patch ayarı veya düşük coreset oranı nedeniyle `embedding_count * ratio < 1` olması. Örneğin 128 embedding ile UI'nin izin verdiği 0.001 oranı.

**Etki:** Tüm embedding'ler çıkarıldıktan sonra coreset oluşturma başarısız olur; kullanıcı işe başlarken ayarın geçersizliğini göremez. O anki başarım veya yavaşlık ölçülmedi.

**Uygulanabilir çözüm:** Eğitim planında tahmini embedding ve coreset adetlerini göstermek; gerçek embedding sayısı belli olduğunda coreset seçimine geçmeden en az bir örnek garantisini açıkça kontrol etmek. Oranı sessizce değiştirmek yerine etkin örnek sayısını raporlamak ve kullanıcıya parametreye ilişkin teşhis vermek.

**Kabul testi:** 128×0.001 kombinasyonu anlaşılır veri/ayar mesajıyla reddedilmeli; 1000×0.001 için 1 örnek sınırı doğru hesaplanmalı. Patch stride ve girdi boyutu değişince tahmin güncellenmeli. Tam model eğitimi gerekmeyen count/ratio sınır testleri yeterlidir.

**Efor:** 0.5–1 gün.

Resmi kaynak sıfır coreset kontrolünü doğruluyor: [Anomalib 2.6.0 KCenterGreedy](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/components/sampling/k_center_greedy.py).

## Olumlu kontroller ve sınırlar

- `create_model`, Anomalib constructor'ında bulunmayan uygulamaya özgü `image_size`, PatchCore `sigma`, `patch_size`, `patch_stride` alanlarını çıkarıyor; bileşenler üzerinden uyguluyor ve parametre snapshot'ını koruyor. EfficientAD `s/m` değerlerini kitaplığın `small/medium` enum değerlerine dönüştürmesi doğru.
- EfficientAD train batch=1 zorlaması kurulu kitaplığın şartına uygun. EfficientAD preprocessor Normalize eklemiyor; normalizasyon kitaplığın forward yoluna bırakılıyor. Pretrained backbone seçimine uygun yerel asset kimliği atanıyor.
- İstatistik uzmanının alanına giren test setinden winner seçme ve bağımsız veri ayrımı bu nota yinelenmedi. Seed/tekrar üretim politikası da ilgili veri/test uzmanlarının alanına bırakıldı.
- Gerçek eğitim, gerçek sonuç checkpoint'i, gerçek model başarımı veya export/inference doğruluğu bu incelemede doğrulanmadı. Başarı oranı, yanlış alarm oranı veya ölçülmüş performans değeri üretilmedi. Model seçiminde veriye dayalı bir winner önerilmedi.

Önerilen uygulama sırası: 03-01 ve 03-02 ile geçersiz parametreleri önceden durdurmak; 03-03 ile etkin kayıt doğruluğunu sağlamak; 03-04 için sürüm/uyarlayıcı kararını sözleşme testiyle doğrulamak; 03-05 ile küçük veri coreset sınırını anlaşılır hale getirmek. Bu analizde çözüm uygulanmadı.
