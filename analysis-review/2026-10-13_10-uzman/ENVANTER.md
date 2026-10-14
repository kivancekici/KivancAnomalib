# Kaynak ve çalışma başlangıcı envanteri

İnceleme tarihi: 13 Ekim 2026 (Europe/Istanbul). Çalışma kökü: `C:\Users\kivanc.ekici\Documents\KivancAnomaLibTest`.

## Yetki ve koruma kapsamı

Kullanıcı analiz ve 10 ayrı yapay zekâ uzman incelemesi istedi. Kaynak kodu ve konfigürasyon düzeltilmedi. Eğitim, büyük veri indirme, paket yükleme, model/veri yükleme veya kamera açma bu incelemenin kapsamına alınmadı. Yeni raporlar ve güvenli analiz çıktıları bu inceleme klasöründe tutulur. Git deposu yoktur; Git işlemi yapılmadı.

Kök ve üst dizinlerde `AGENTS.md`, ayrıca proje ağacında (ortam/build hariç) uygulanabilir alt talimat dosyası bulunmadı. README ve eski raporlar yalnız mevcut durumun kanıtı olarak okundu.

## Teknoloji ve kaynaklar

- Qt Widgets/C++17, qmake; `KivancAnomaLibTest.pro` Qt widgets ve concurrent modüllerini tanımlar. Basler Pylon + OpenCV yerel kamera hattı SDK başlıkları bulununca derlemeye eklenir.
- `mainwindow.cpp`: 4.190 satır; operatör akışı, eğitim/inceleme/ayar sayfaları ve dosya yönetimi.
- `worker/anomalib_worker.py`: 3.487 satır, 90 üst seviye fonksiyon ve 7 sınıf; JSON Lines protokolü, varlık yönetimi, eğitim, dışa aktarma ve kalıcı inference.
- `anomalibprocess.cpp/.h`: Qt QProcess köprüsü; `cameraservice.cpp/.h`: Basler/OpenCV kamera hizmeti.
- `tests/test_worker.py`: 37 test metodu. README'deki 26 sayısı güncel değildir.
- `assets/`: dört tasarım/gösterim PNG'si. `qa/`: dört geçmiş görsel karşılaştırma PNG'si; `design-qa.md` geçmiş tasarım ve kamera smoke-test beyanları içerir. Bunlar bu incelemede tekrar yapılmış test değildir.
- `requirements-inference.txt`: yalnız `onnxruntime-gpu>=1.27,<1.30`; tam eğitim/uygulama ortamının sürüm kilidi değildir.
- `.venv` mevcut. Kurulu sürümler `kanitlar/ortam-surumleri.json` içinde saklandı: Python 3.12.14; Anomalib 2.6.0; Torch 2.13.0+cu130; TorchVision 0.28.0+cu130; Lightning 2.6.5; timm 1.0.28; OpenVINO 2026.2.1; ONNX 1.22.0; onnxruntime-gpu 1.28.0; NumPy 2.4.6; Pillow 12.3.0; scikit-learn 1.9.0. `onnxruntime` dağıtım adı ayrı kurulu değildir; `onnxruntime-gpu` aynı Python modülünü sağlar.

## Baseline ve mevcut veri/sonuçlar

`kanitlar/kaynak-baseline.json`, 25 kaynak/belge/test/asset/QA dosyasının mutlak yolu, boyutu, UTC değişiklik zamanı ve SHA-256 değerini içerir. Bu küçük dosyaların içerik bütünlüğü inceleme sonunda yeniden kontrol edilir.

`kanitlar/veri-build-envanter.json`, başlangıçta `.data` ve `build` altındaki 2.550 dosyanın yol, boyut ve UTC değişiklik zamanını içerir. Büyük veri ve model ikililerine başlangıçta hash taraması uygulanmadı; bu dosyalar için metadata koruma kontrolü yapılır. Bu kontrol içerik hash doğrulamasıyla eşdeğer değildir.

- Kök `.data`: 11 dosya; tek kamera smoke-test görüntüsü, iki backbone safetensors, iki EfficientAD teacher ağırlığı, assets-lock ve yarım ImageNette indirmesi dahil. Smoke-test görüntüsü üretim değerlendirme veri kümesi sayılmaz.
- `build`: 2.539 dosya; dört native EXE, farklı build dizinlerine dağılan iş JSON'ları, manifestler, görüntüler ve bir ONNX teslim dosyası dahil. Dört build ağacı mevcut olması güncel derlemenin bu incelemede başarıyla yapıldığı anlamına gelmez.
- Envanterde `.ckpt` veya `.pt` dosyası yok. ONNX dosyası `build/Desktop_Qt_6_11_1_MSVC2022_64bit_Release/release/.data/onnx-delivery/dataset3-patchcore-v13.onnx` (607.008.673 bayt). Bu büyük dosya analizin başlangıcında modele yüklenmedi.
- Mevcut `kernel.errors.txt` geçmiş CISA kernel hata metnidir; tarih/iş/model bağlantısı olmadan güncel çalışma başarısızlığı sayılmaz.

## Gerçekten çalıştırılan ana kontroller

1. Dosya/talimat araması, AST üzerinden worker/test envanteri ve kurulu paket metadata okuması. AST okuması bytecode veya kaynak değişikliği oluşturmadı.
2. `.venv/Scripts/python.exe -B -X utf8 -m unittest discover -s tests -v`: ilk koşu 37 test, 29 ERROR; Windows sandbox varsayılan geçici dizininde PermissionError. Log: `kanitlar/unittest.log`, sonuç: `kanitlar/unittest-sonuc.json`.
3. Aynı suite, TEMP/TMP bu inceleme klasöründeki `kanitlar/test-temp` ile yeniden çalıştı: **37 test OK, 67,930 saniye**, exit code 0. Log: `kanitlar/unittest-workspace-temp.log`; komut ve duvar saati: `kanitlar/unittest-workspace-temp-sonuc.json`. `PYTHONDONTWRITEBYTECODE=1`, UTF-8 ve HF offline bayrakları alt süreçlere aktarıldı. Ağ indirme testleri mock/fake yanıtlarla çalışır; gerçek indirme yapılmadı. Gerçek model constructor kontrolleri ağırlık indirmeden çalışır; bunlar eğitim değildir.

4. Worker `run_check()` gerçek ortamı salt okunur keşfetti: `ok=true`; CUDA RTX 3060 Laptop GPU; ONNX Runtime provider listesi TensorRT/CUDA/CPU; OpenVINO CPU, Intel iGPU ve NVIDIA dGPU bildirildi. `kanitlar/ortam-check.log` kaydı vardır. Provider listelenmesi, gerçek bir modelde başarılı GPU yürütmesi veya gecikme ölçümü anlamına gelmez.

Native uygulama bu incelemede başlatılmadı, native build tekrarlanmadı ve kamera açılmadı. Model doğruluk/yanlış alarm oranı için yeni üretim veri ölçümü yapılmadı. Uzmanların ek hafif kontrolleri kendi notlarında ve ana raporda ayrı belirtilir.

## Tamamlanma ve son koruma kontrolü

10 ayrı uzman görevinin tamamlanma yanıtı alındı; 10 bağımsız not mevcut. 53 özgün bulgu ana raporda 43 numaralı iş kalemine eşlendi. [Ana rapor](ANA_RAPOR.md), ilk 9 iyileştirmeyi, her bulgunun tür/kanıt/tetikleyici/etki/çözüm/kabul/efor alanlarını, uygulama sırasını ve açık doğrulamaları içerir. [Uzman görev kaydı](uzman-is-kayitlari.json) ayrı agent görevlerini, tamamlanma durumlarını ve not hash'lerini saklar.

Ek kontroller salt okunur paket/resource metadata, yeni küçük görüntü/dosya fixtures, sayısal/AST kontrolleri, fake backend/clock ve küçük ONNX graph'larının yalnız CPU'da yürütülmesidir. Üretim model/veri dosyaları üzerinde eğitim veya inference yapılmadı. `pip check` exit 0 ile temiz. Bütün yeni kontrollerin kapsam/sınırı uzman notlarında belirtilir.

[Son baseline kontrolü](kanitlar/baseline-son-kontrol.json) exit 0: 25 kaynak dosyasının SHA-256/boyut/mtime'ı aynı; `.data/build` altındaki 2.550 dosyanın boyut/mtime'ı aynı, eksilen veya yeni dosya yok. Büyük ikililer içerik hash'iyle doğrulanmadı. Bu kontrol `.venv` veya baseline dışındaki bütün dosyalar için kapsamlı içerik bütünlüğü garantisi değildir.

[Rapor kontrolü](kanitlar/rapor-son-kontrol.json) exit 0: B01–B43 sıralı ve gerekli alanları dolu; 53 uzman bulgusunun tümü birer kez eşlenmiş; 10 not ve 35 yerel ana rapor bağlantısı mevcut; UTF-8 replacement character yok. Uygulama düzeltmesi yapılmadı.
