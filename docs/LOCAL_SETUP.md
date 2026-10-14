# Yeni depodan çalışma

Proje kökü: **C:\Users\kivanc.ekici\Documents\GitHub\KivancAnomalib**.
Qt proje dosyasının ve çalıştırılabilir dosyanın adı uyumluluk için
KivancAnomaLibTest olarak korunmuştur. Kod, testler, kaynak görselleri ve
analysis-review/ altındaki eski raporlar aynı göreli yerleşimdedir.
Ana özellik ve kullanım açıklamaları [README](../README.md) içindedir.

## Qt Creator ve derleme

Qt Creator'da bu depodaki **KivancAnomaLibTest.pro** dosyasını açın ve
**Qt 6.11.1 / MSVC 2022 64 bit** kitini seçin. Yeni bir build/ alt klasörü
kullanın. Eski kişisel Qt Creator ayarları ve derleme ürünleri aktarılmadı.
Basler ve OpenCV'nin bu makinedeki SDK yolları .pro dosyasında korundu.
Başka makinede SDK konumlarını uyarlayın; SDK'lar bulunmazsa kamera desteği
olmayan uygulama derlenebilir.

Alternatif olarak **x64 Native Tools Command Prompt for VS 2022** içinde:

~~~bat
cd /d C:\Users\kivanc.ekici\Documents\GitHub\KivancAnomalib
mkdir build\desktop-debug
cd build\desktop-debug
C:\Qt\6.11.1\msvc2022_64\bin\qmake.exe ..\..\KivancAnomaLibTest.pro -spec win32-msvc CONFIG+=debug
nmake
set PATH=C:\Qt\6.11.1\msvc2022_64\bin;C:\Program Files\Basler\pylon\Runtime\x64;C:\opencv\build\x64\vc16\bin;%PATH%
debug\KivancAnomaLibTest.exe
~~~

## Python ortamı

4 GiB kurulu .venv kopyalanmadı. Hemen devam etmek için Settings sayfasında
mevcut **C:\Users\kivanc.ekici\Documents\KivancAnomaLibTest\.venv\Scripts\python.exe**
dosyasını seçebilirsiniz. Bu, eski ortama geçici bir bağımlılıktır; eski
klasörü silmeyin. Proje kökü araması artık önce yeni uygulamanın konumundan
başlar, böylece bu Python seçimi yerel model varlıklarını eski projeye yöneltmez.
Önceden kaydedilmiş veri/model yollarını Settings/Train/Inspect içinde yeniden
seçin; Windows kullanıcı ayarları taşımada değiştirilmedi.

Bağımsız yeni ortam için Python 3.12 ile proje kökünde aşağıdaki komutları
kullanabilirsiniz. Taşıma sırasında yeni ortam kurulumu yapılmadı.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:ANOMALIB_PYTHON = "$PWD\.venv\Scripts\python.exe"
~~~

Buradaki python komutunun Python 3.12 çalıştırdığını önceden kontrol edin.
requirements.txt kaynak ortamda gözlenen ana bağımlılık sürümlerini belirtir.
Kaynakta PyTorch 2.13.0+cu130 / torchvision 0.28.0+cu130 vardı; yeni
kurulumda CPU/CUDA tekerleği ve sürücü uyumunu ayrıca doğrulayın. İsteğe bağlı
CUDA 13 ONNX sağlayıcısı için mevcut requirements-inference.txt kullanılabilir.
Tam kaynak paket listesi docs/migration/requirements-source-environment.txt
dosyasında referans olarak saklanır; başka makinenin kurulum garantisi değildir.

Uygulamada **Verify environment** çalıştırın. İlk UI denemesinde
**Use demonstration results** açık tutulabilir. Gerçek eğitim/inference için
uygun veri kümesini ve hazır model varlıklarını doğrulayın.

## Yerel veri ve modeller

Git dışında kalan .data/ klasörü bu çalışma kopyasında fiziksel olarak vardır:

- .data/model-assets/: PatchCore/PaDiM backbone ağırlıkları, EfficientAD
  öğretmen ağırlıkları ve konumu yeni köke uyarlanmış assets-lock.json.
- .data/camera-smoke-test/: önceki gerçek kamera örneği.
- .data/migrated-builds/&lt;kit&gt;/&lt;debug veya release&gt;/: eski derlemelerin
  çalışma verileri, job JSON'ları, inference görselleri ve model teslimi.
- .data/migration-diagnostics/: eski yerel hata kaydı.

Korunan ONNX modeli Inspect → **Load model…** ile şu manifestten seçilebilir:

~~~text
.data/migrated-builds/Desktop_Qt_6_11_1_MSVC2022_64bit_Release/release/onnx-delivery/dataset3-patchcore-v13.manifest.json
~~~

Manifestin artifact.path alanı göreli olduğu için ONNX dosyası yeni konumunda
çözülebilir. source_checkpoint tarihsel bir kayıttır; işaret ettiği .ckpt
kaynak klasörde de bulunmamıştır. ONNX teslimi korunmuştur, orijinal eğitim
checkpoint'inin geri getirildiği varsayılmamalıdır. Diğer eski job/production
JSON'ları tarihsel kayıtlardır; dış veri kümesi yollarını otomatik tekrar
çalıştırmadan önce yeniden seçin. Masaüstündeki veri kümeleri bu projenin
içeriği değildir ve kopyalanmadı.

Eksik ImageNette indirmesi (imagenette2.tgz.part) ve aktarım logları kaynakta
kaldı. EfficientAD için tamamlanmış ImageNette verisi bu kaynakta yoktur;
gerekiyorsa mevcut uygulama akışından açıkça hazırlanmalıdır.

Yeni uygulamanın job ve inference çıktıları yürütülebilir dosyanın yanındaki
.data/ klasörüne yazılır. Model varlıkları proje kökündeki .data/model-assets
altında bulunur. Taşınan eski sonuçlar otomatik yeni çalışma sonuçları sayılmaz.

.data/, ortamlar ve derlemeler .gitignore ile dışarıda tutulur. Git clone
veya push bunları taşımaz; yerel model/veri yedeğini ayrıca koruyun. Küçük
test/probe örnekleri ve QA görselleri Git kapsamında kalır.

## Düşük maliyetli doğrulama

Yeni proje kökünden mevcut Python ortamıyla test çalıştırmak için:

~~~powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\kivanc.ekici\Documents\KivancAnomaLibTest\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
~~~

Testler taşınan worker'ı yeni proje kökünden yükler. Önceki
analysis-review/2026-10-13_10-uzman/kanitlar/ kayıtları tarihsel kanıttır;
içlerindeki eski mutlak yollar ve ortam sürümleri o incelemeyi temsil eder.
Taşımanın güncel kontrolleri [taşıma raporunda](migration/2026-10-14.md) yer alır.
