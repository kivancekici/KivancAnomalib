# Uzman 04 — Anomali skorları, eşik kalibrasyonu, metrikler ve istatistik

İnceleme: 13 Ekim 2026. Ayrı yapay zekâ uzman incelemesi; kaynak ve konfigürasyon değişikliği uygulanmadı.

## Kapsam ve karar

Worker `metric_value`, `normalized_map`, tahmin/inspection artefaktları, gerçek eğitim sonucu ve winner seçimi, gerçek inference ve export postprocessor yolları; Qt sonuç, karar eşiği ve maske gösterim kodları incelendi. Kurulu Anomalib **2.6.0**, NumPy **2.4.6** ve Pillow **12.3.0** sürümleri mevcut ortam envanterinden ve yerel paket kaynağından doğrulandı. Kütüphane davranışı resmi **lib/v2.6.0** kaynak etiketiyle karşılaştırıldı.

En yüksek öncelik, modelden gelen piksel kalibrasyonunun görüntü içi min–max tarafından kaybedilmesi ve eksik/geçersiz skorun karar olarak kullanılabilmesidir. Sentetik örnekler bu yazılım davranışlarını doğruladı; gerçek veri üzerinde yanlış alarm/kaçırma oranı veya başarı yüzdesi ölçülmedi. Üretim datasetleri erişilebilir olmadığı için eşik önerisini belirli bir yüzdeye sabitlemek mümkün değildir.

## Bulgular

### M04-01 — Görüntü içi min–max, kalibre edilmiş piksel haritasını ve eşiğin anlamını bozuyor

- **Tür / önem:** Doğrulanmış hata; **P1**. Maske/heatmap kararının güvenilirliği etkilenir.
- **Kanıt:** [normalized_map](../../../worker/anomalib_worker.py), minimumu her görüntüden çıkarıp kendi aralığına bölüyor; sabit haritayı tümüyle sıfırlıyor. [write_prediction_artifacts](../../../worker/anomalib_worker.py), bu haritadan [1891](../../../worker/anomalib_worker.py) maskesini üretiyor. Aynı işlemi eğitim inspection yolu [1939](../../../worker/anomalib_worker.py) yapıyor. UI, modelin `pred_mask` dosyası bulunsa dahi [1674](../../../mainwindow.cpp) ve [3932](../../../mainwindow.cpp) üzerinden min–max PNG'ye yeniden eşik uyguluyor. Yerel PostProcessor: `.venv/Lib/site-packages/anomalib/post_processing/post_processor.py:219–228, 302–324`; modelin doğrulama kümesinden öğrendiği normalizasyon/eşik zaten uygulanmış durumda.
- **Çalışan kanıt:** 2×2 harita `[0.10,0.13;0.17,0.20]`, %60 eşikte 0/4 yerine **2/4 kırmızı** piksel üretti. `[0.80,0.83;0.87,0.90]` aynı maskeye dönüştü ve 4/4 yerine **2/4** kaldı. Sabit `0.90` harita 4/4 yerine **0/4** oldu. [Probe sonucu](../kanitlar/probe-04-metrics-result.json).
- **Tetikleyici:** Kalibre edilmiş haritası dar, düşük veya yüksek aralıkta kalan görüntü; tüm pikselleri aynı yüksek değere clip edilmiş bir çıktı.
- **Etki:** Düşük harita normal görüntüde görünür kusur bölgesi varmış izlenimi yaratabilir; yüksek harita kusur alanını küçültebilir, sabit yüksek harita kusuru görsel olarak silebilir. Bu bulgu **görüntü düzeyindeki `pred_score` sınıflandırmasını doğrudan değiştirmez**; yanlış alarm ve kaçırma etkisi maske/operatör incelemesi düzeyindedir. Gerçek hata oranları bilinmiyor.
- **Çözüm:** Karar haritasının kalibre edilmiş sayısal ölçeğini korumak; görsel kontrast ölçeklemesi gerekirse yalnız heatmap görüntüsüne uygulayıp karar haritasından ayrı tutmak. Kalibre edilmiş map veya model `pred_mask` aynı eşik sözleşmesiyle kullanılmalı. Ham sigma haritası dönen export yolları ayrıca doğrulama kalibrasyonuyla işlenmeli. Float harita korunmalı; 8 bit PNG kararın tek kaynağı olmamalı.
- **Kabul testi:** Yukarıdaki üç haritada %60 maskesi sırasıyla 0/4, 4/4, 4/4 olmalı. Bir görüntüye başka parlak piksel eklemek mevcut bir pikselin kalibre edilmiş değerini/eşik kararını değiştirmemeli. Aynı görüntü ve eşik için checkpoint/ONNX/OpenVINO maske kararları açıklanmış toleransla aynı olmalı.
- **Efor:** 1–2 gün; export sözleşmesinin hizalanması uzman 06 bulgularıyla birleştirilebilir.

### M04-02 — Eksik/geçersiz görüntü skoru güvenilir tahmin olarak kullanılabiliyor

- **Tür / önem:** Doğrulanmış hata; **P1**.
- **Kanıt:** [scalar_value](../../../worker/anomalib_worker.py) finiteness doğrulamıyor. [1899–1905](../../../worker/anomalib_worker.py), skor eksikse min–max yapılmış haritanın maksimumuna dönüyor; NaN karşılaştırması false olduğu için iç sonuç etiketi `normal` oluyor.
- **Çalışan kanıt:** Normal aralıktaki `[0.10..0.20]` harita ve eksik `pred_score` ile çıktı **score=1.0, anomalous**. Yüksek `[0.80..0.90]` harita ve `pred_score=NaN` ile iç sonuç **score=NaN, normal**. [Probe sonucu](../kanitlar/probe-04-metrics-result.json). NaN sonrasında JSON protokolü de bozulabilir; **UI'da mutlaka NORMAL gösterildiği iddia edilmez** (M04-03).
- **Tetikleyici:** Backend'in görüntü skoru üretmemesi, yanlış çıktı alanı eşlemesi veya sayısal taşma/NaN. Gerçek modelde bu tetikleyicinin gerçekleştiği gözlemlenmedi.
- **Etki:** Eksik skor, neredeyse her sabit olmayan harita için görüntü anomali skorunu 1'e zorlayıp yanlış alarm üretir. NaN'ın `normal` iç etiketi oluşturması ise fail-safe karar sözleşmesini ihlal eder; geçersiz hesap başarısızlık olarak ayrılmalıdır.
- **Çözüm:** Skorun tek görüntüye ait, finite ve sözleşmeye uygun aralıkta olduğunu doğrulamak. Eksik skor için ancak backend'in dokümante edilmiş **ham** pikselden görüntü skoru kuralı ve gerekli kalibrasyon varsa geri dönüş yapılmalı; aksi halde açık `prediction_invalid` hatası döndürülmeli. `pred_label`/manuel eşik sözleşmesi ayrıca M04-04 kapsamında tanımlanmalı.
- **Kabul testi:** None/NaN/+Inf/−Inf/skoru boş dizi olan çıktılar asla normal/anomalous başarı olayı üretmemeli. Son geçerli operator kararı yeni, geçersiz tahminmiş gibi sunulmamalı. Eksik skorun ham-map fallback'i destekleniyorsa bağımsız bir fixture'da referans model kuralına eşit olmalı.
- **Efor:** 0.5–1 gün.

### M04-03 — Geçersiz/eksik metrikler JSON olayını ve kazanan seçimini bozuyor

- **Tür / önem:** Doğrulanmış hata; **P2**.
- **Kanıt:** [metric_value](../../../worker/anomalib_worker.py), NaN/Inf'i float olarak kabul ediyor. [emit](../../../worker/anomalib_worker.py) `json.dumps` varsayılanıyla standart JSON olmayan `NaN`/`Infinity` sabitlerini yazabilir. Qt [consumeOutputLines](../../../anomalibprocess.cpp), geçerli JSON olmayan satırı event yerine düz çıktı olarak ele alıyor. [winner seçimi](../../../worker/anomalib_worker.py), yalnız None değerini dışlar; tüm metrikler eksikse ilk modeli seçer. UI [updateWinner](../../../mainwindow.cpp) bunu koşulsuz Winner olarak sunar.
- **Çalışan kanıt:** NaN AUROC, `{"event":"result",..."image_auroc":NaN}` içerikli olay üretti; strict JSON parser reddetti. Tüm AUROC değerleri None iken PatchCore winner seçildi. İlk skor NaN, ikinci PaDiM skoru 0.99 iken `max` yine PatchCore seçti. [Probe sonucu](../kanitlar/probe-04-metrics-result.json). Qt parse/işaretleme etkisi kaynak üzerinden incelendi; native GUI çalıştırılmadı.
- **Tetikleyici:** Yetersiz/tek sınıflı değerlendirme sonucu geçersiz metrik, değişmiş/eksik metrik alanı veya backend'in NaN üretmesi. Her tek sınıflı kümenin mutlaka NaN verdiği iddia edilmez; kütüphane bazı metriklerde 0+uyarı verebilir.
- **Etki:** Sayısal sonuç olayı UI'ya ulaşmayabilir; hiçbir karşılaştırılabilir skor yokken veya geçersiz skor öne geçerken model kazanan ilan edilebilir. Operatör doğrulanmış üstünlük varmış gibi export/seçim yapabilir.
- **Çözüm:** Metrikleri finite ve [0,1] aralığında doğrulamak; eksik/geçersizi `null` ve açıklanan durumla standart JSON olarak göndermek (`allow_nan=False`, önce payload temizleme). Kazanan seçimini iki sınıflı geçerli değerlendirme ve aynı manifest koşuluna bağlamak; skor yoksa winner olmamalı. Aynı skorlarda deterministik tie kuralı ve belirsizlik gösterimi tanımlanmalı.
- **Kabul testi:** NaN/Inf/None veya tümü eksik metrik olayları standart JSON olarak parse edilmeli; geçerli model geçersiz skordan dolayı kaybetmemeli; tüm skorlar geçersizse `winner` boş/None ve anlaşılır “karşılaştırma yok” durumu gösterilmeli.
- **Efor:** 0.5–1 gün; dayanıklılık/protokol incelemesiyle birleştirilebilir.

### M04-04 — Gösterilen F1 ile inference karar eşiği farklı çalışma noktalarını temsil ediyor

- **Tür / önem:** Koşullu risk; **P2**. Manuel eşik değiştirme tek başına hata değildir; risk metrikle çalışma noktası ilişkisinin görünmez olmasıdır.
- **Kanıt:** Anomalib varsayılan PostProcessor sensitivities=0.5; yerel kaynak `.venv/Lib/site-packages/anomalib/post_processing/post_processor.py:66–67, 351–366`, varsayılan evaluator F1'i `pred_label` üzerinden hesaplıyor (`models/components/base/anomalib_module.py:378–382`). Worker [image_f1](../../../worker/anomalib_worker.py) bu değeri sunuyor. Inference UI [1871](../../../mainwindow.cpp) eşiği %60 ile açıyor; worker [1905](../../../worker/anomalib_worker.py) model `pred_label` değerini kullanmayıp seçilen eşiğe göre karar veriyor. Sonuç tablosu [2988](../../../mainwindow.cpp) ve inference mesajı [3656](../../../mainwindow.cpp) bu farklı çalışma noktasına ait F1/FPR/FNR açıklamıyor.
- **Çalışan kanıt:** `pred_score=0.55, pred_label=True` girdisi inference %60 eşiğinde **normal** oldu. Bu sentetik davranış doğrulamasıdır; gerçek hatalı ürün kaçırma oranı değildir.
- **Tetikleyici:** Operatör sonuç tablosundaki iyi F1'i varsayılan %60 çalışma noktasının kalitesi sanarak devreye alır veya deployment eşiğini değiştirir.
- **Etki:** Varsayılan .5'ten .6'ya çıkmak .5–.6 bandındaki pozitif kararları negatife çevirir; gerçek anomaliler bu bandaysa kaçırma artabilir, normal örnekler bu bandaysa yanlış alarm azalabilir. Anomalib normalize skoru kusur olasılığı değildir. Kalibre edilmiş formülde .60, ham eşikten `0.10 × (validation_max−validation_min)` kadar yukarıdaki karar noktasını temsil eder.
- **Çözüm:** Modelin kalibrasyon/çalışma eşiğini artefakt manifestinde taşımak; varsayılan inference eşiğini bu sözleşmeden almak veya override'i açıkça belirtmek. Metrik raporuna kullanıldığı eşik, validation/test sınıf sayıları, TP/FP/TN/FN, precision/recall ve FPR/FNR eklemek. Eşik seçimi validation kümesinde veya bağımsız kalibrasyon kümesinde yapılmalı; seçilen noktadaki başarı untouched holdout'ta ölçülmeli.
- **Kabul testi:** .55 skorlu örneğin model ve manuel-eşik kararları ayrı kaydedilmeli. Sonuç raporu hangi eşikte F1 hesaplandığını göstermeli. Deployment eşiği değişince eski F1 yeni eşikte ölçülmüş gibi sunulmamalı; hedef FPR/recall kabulü etiketli, bağımsız veriyle yapılmalı.
- **Efor:** 1–2 gün yazılım; istatistiksel doğrulama süresi etiketli veriye bağlıdır.

### M04-05 — Modellerin karşılaştırması ortak değerlendirme ayrımına sabitlenmemiş

- **Tür / önem:** Doğrulanmış hata (ayrım/karşılaştırma sözleşmesi); **P1**. Aynı gerçek veride kazananın ne kadar değiştiği ölçülmedi.
- **Bağımsız statik kanıt:** [create_datamodule](../../../worker/anomalib_worker.py) seed/split manifesti geçirmiyor; [2267](../../../worker/anomalib_worker.py) her model için yeniden oluşturuyor ve [2327](../../../worker/anomalib_worker.py) bu skorları maksimumla karşılaştırıyor. Kaynakta global seed ayarı da bulunmadı. Kurulu Folder `seed=None`, `val_split_mode=FROM_TEST`, `val_split_ratio=0.5` varsayılanı (`data/datamodules/image/folder.py:132–136`), ayrım `random_split` (`data/datamodules/base/image.py:344–350`) ve `torch.randperm` (`data/utils/split.py:176–177`) kullanıyor.
- **Tetikleyici:** Aynı işte birden fazla model veya yeniden yapılan denemeler.
- **Etki:** AUROC farkı model etkisiyle örnek seçimi etkisini karıştırır. Aynı test üzerinde tekrar tekrar en iyi modeli/parametreyi seçmek de son başarı tahminini iyimserleştirebilir. Bu durum dosya yolu düzeyinde train/test sızıntısının kanıtı değildir: **FROM_TEST aynı çalıştırma içinde validation ve testten ayrık alt kümeler oluşturur**.
- **Çözüm:** İş başında sabit dosya/grup ayrımı manifesti; modellerin aynı örneklerde tahmini; model/eşik seçiminden sonra ayrıca untouched holdout. Karşılaştırma gerektiğinde ortak örneklerden paired bootstrap güven aralığı veya kusur/lot bazlı tekrarlar; az anomalili durumda tek AUROC ondalığı kesin üstünlük olarak sunulmamalı.
- **Kabul testi:** Tüm modellerin test ve validation manifest ID'leri eşit olmalı; tekrar koşu aynı ayrımı üretmeli; hiçbir splitte grup/hash çakışması olmamalı; final holdout model/eşik seçimine hiç katılmamalı.
- **Efor:** 1–2 gün yazılım; ek veri/istatistik aşaması veri erişimine bağlıdır.
- **Tekrar bilgisi:** Uzman 02 V02-01 ile aynı temel bulgudur; ana raporda tek bulguya birleştirilmeli. Bu rol veri uzmanının metadata deneyini yeniden çalıştırmadı; burada bağımsız kaynak izleme kanıtı vardır.

## Gerçekten yürütülen kontroller ve sınırlar

1. Worker/Qt ve kurulu Anomalib postprocessor, evaluator, F1AdaptiveThreshold, Folder ve split kodları statik okundu. Resmi sürüm kaynağı web aracından okundu; ilk yanlış `v2.6.0`/eski repo URL denemeleri erişmedi, doğru `open-edge-platform/anomalib/lib/v2.6.0` kaynakları erişti.
2. `.venv/Scripts/python.exe -B -X utf8 analysis-review/2026-10-13_10-uzman/kanitlar/probe_04_metrics.py` yürütüldü; `PYTHONDONTWRITEBYTECODE=1`, çıkış 0. Betik yalnız worker'ın saf yardımcı fonksiyonlarıyla küçük NumPy dizileri ve 2×2 Pillow dosyaları kullandı. Model yükleme, eğitim, gerçek inference, GPU işi, kamera veya ağ aktarımı yapılmadı. Yeni çıktılar yalnız review klasöründedir.
3. [Betik](../kanitlar/probe_04_metrics.py) ve [sonuç](../kanitlar/probe-04-metrics-result.json) bağımsız yeniden üretilebilir kanıtlardır. `kanitlar/probe-04/` içindeki dosyalar sentetik artefaktlardır.
4. Ana ajanın **37 test OK / 67,930 saniye** kontrolü `kanitlar/unittest-workspace-temp.log` kaydından bilinir; bu rol suite'i tekrar çalıştırmadı. Mevcut testlerin geçmesi yukarıdaki karar/kalibrasyon durumlarını kapsadıkları anlamına gelmez.
5. Gerçek dataset sınıf/maske sayıları, ROC/PR eğrisi, FPR/FNR, güven aralığı veya optimum üretim eşiği ölçülmedi. Native GUI ve parser başlatılmadı. 607 MB ONNX dosyası yüklenmedi. NaN/eksik skor örneklerinin gerçek modellerde gerçekleşme sıklığı bilinmiyor.

## Resmi sürüm doğrulama kaynakları

Resmi [Anomalib lib/v2.6.0 PostProcessor kaynağı](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/post_processing/post_processor.py), doğrulama istatistiği/eşiğini kullanan normalizasyonu ve varsayılan .5 sensitivity değerlerini doğrular. [lib/v2.6.0 evaluator kaynağı](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/models/components/base/anomalib_module.py) F1'in tahmin etiketi alanını kullandığını gösterir. [lib/v2.6.0 Folder kaynağı](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/data/datamodules/image/folder.py) split/seed varsayılanlarını doğrular. [lib/v2.6.0 F1AdaptiveThreshold kaynağı](https://raw.githubusercontent.com/open-edge-platform/anomalib/lib/v2.6.0/src/anomalib/metrics/threshold/f1_adaptive_threshold.py), tek sınıflı validation için fallback eşik davranışını açıklar. Sürümlü [postprocessing rehberi](https://anomalib.readthedocs.io/en/lib-v2.6.0/markdown/guides/how_to/models/post_processor.html) de kontrol edildi; sürüme bağlı kesin kararlar resmi etiket ve kurulu kaynak üzerinden verildi.

Önerilen uygulama sırası: M04-02 sayısal geçerlilik ve M04-03 JSON/winner koruması → M04-01 karar haritası ölçeği → M04-05 ortak değerlendirme manifesti → M04-04 eşik/operating-point raporlama → gerçek veriyle son holdout kabulü. Uygulama bu analiz talebine dahil değildir.
