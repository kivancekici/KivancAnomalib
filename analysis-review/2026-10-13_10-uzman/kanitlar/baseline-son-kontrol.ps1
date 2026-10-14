$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\kivanc.ekici\Documents\KivancAnomaLibTest'
$reviewRoot = Join-Path $taskRoot 'analysis-review\2026-10-13_10-uzman'
$evidenceRoot = Join-Path $reviewRoot 'kanitlar'
$sourceBaseline = @(Get-Content -LiteralPath (Join-Path $evidenceRoot 'kaynak-baseline.json') -Raw -Encoding UTF8 | ConvertFrom-Json)
$dataBaseline = @(Get-Content -LiteralPath (Join-Path $evidenceRoot 'veri-build-envanter.json') -Raw -Encoding UTF8 | ConvertFrom-Json)
$sourceDiffs = @()
foreach ($entry in $sourceBaseline) {
    if (-not (Test-Path -LiteralPath $entry.path -PathType Leaf)) {
        $sourceDiffs += [pscustomobject]@{ path = $entry.path; reason = 'missing' }
        continue
    }
    $currentFile = Get-Item -LiteralPath $entry.path
    $currentHash = (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash
    $previousTime = if ($entry.modifiedUtc -is [DateTime]) { $entry.modifiedUtc.ToUniversalTime() } else { [DateTimeOffset]::Parse($entry.modifiedUtc).UtcDateTime }
    if ($currentHash -ne $entry.sha256 -or $currentFile.Length -ne $entry.bytes -or $currentFile.LastWriteTimeUtc.Ticks -ne $previousTime.Ticks) {
        $sourceDiffs += [pscustomobject]@{ path = $entry.path; reason = 'hash_or_metadata_changed'; currentSha256 = $currentHash }
    }
}
$dataDiffs = @()
$knownPaths = @{}
foreach ($entry in $dataBaseline) {
    $knownPaths[$entry.path.ToLowerInvariant()] = $true
    if (-not (Test-Path -LiteralPath $entry.path -PathType Leaf)) {
        $dataDiffs += [pscustomobject]@{ path = $entry.path; reason = 'missing' }
        continue
    }
    $currentFile = Get-Item -LiteralPath $entry.path
    $previousTime = if ($entry.modifiedUtc -is [DateTime]) { $entry.modifiedUtc.ToUniversalTime() } else { [DateTimeOffset]::Parse($entry.modifiedUtc).UtcDateTime }
    if ($currentFile.Length -ne $entry.bytes -or $currentFile.LastWriteTimeUtc.Ticks -ne $previousTime.Ticks) {
        $dataDiffs += [pscustomobject]@{ path = $entry.path; reason = 'size_or_modified_time_changed' }
    }
}
$currentDataFiles = @(Get-ChildItem -LiteralPath (Join-Path $taskRoot '.data'), (Join-Path $taskRoot 'build') -Recurse -Force -File)
$newDataPaths = @($currentDataFiles | Where-Object { -not $knownPaths.ContainsKey($_.FullName.ToLowerInvariant()) } | ForEach-Object { $_.FullName })
$roster = @(
    @{ task = '/root/uzman_01_mimari'; title = 'Mimari ve durum yönetimi'; note = '01-mimari.md' },
    @{ task = '/root/uzman_02_veri'; title = 'Veri ve değerlendirme ayrımı'; note = '02-veri.md' },
    @{ task = '/root/uzman_03_egitim'; title = 'Eğitim ve model parametreleri'; note = '03-egitim.md' },
    @{ task = '/root/uzman_04_metrikler'; title = 'Metrikler ve karar kalibrasyonu'; note = '04-metrikler.md' },
    @{ task = '/root/uzman_05_goruntu'; title = 'Görüntü işleme ve koordinatlar'; note = '05-goruntu.md' },
    @{ task = '/root/uzman_06_deployment'; title = 'Export ve deployment'; note = '06-deployment.md' },
    @{ task = '/root/uzman_07_performans'; title = 'Performans ve kapasite'; note = '07-performans.md' },
    @{ task = '/root/uzman_08_dayaniklilik'; title = 'Dayanıklılık ve süreç ömrü'; note = '08-dayaniklilik.md' },
    @{ task = '/root/uzman_09_operator'; title = 'Operatör akışı'; note = '09-operator.md' },
    @{ task = '/root/uzman_10_test_surum'; title = 'Testler ve sürümleme'; note = '10-test-surum.md' }
)
$noteRecords = @()
$findingTotal = 0
foreach ($expert in $roster) {
    $notePath = Join-Path (Join-Path $reviewRoot 'uzman-notlari') $expert.note
    if (-not (Test-Path -LiteralPath $notePath -PathType Leaf)) { throw "Expert note missing: $notePath" }
    $content = Get-Content -LiteralPath $notePath -Raw -Encoding UTF8
    $findingCount = [regex]::Matches($content, '(?m)^#{2,3} (?:M\d{2}|V\d{2}-\d{2}|M\d{2}-\d{2}|D\d{2}-\d{2}|T\d{2}-\d{2}|\d{2}-\d{2}) ').Count
    $findingTotal += $findingCount
    $noteRecords += [pscustomobject]@{ task = $expert.task; title = $expert.title; status = 'completed'; completionEvidence = 'separate agent final response received by root'; note = $notePath; bytes = (Get-Item -LiteralPath $notePath).Length; sha256 = (Get-FileHash -LiteralPath $notePath -Algorithm SHA256).Hash; findings = $findingCount }
}
$record = [pscustomobject]@{
    checkedUtc = [DateTime]::UtcNow.ToString('o')
    sourceFilesChecked = $sourceBaseline.Count
    sourceHashAndMetadataDiffs = @($sourceDiffs)
    dataBuildFilesChecked = $dataBaseline.Count
    currentDataBuildFileCount = $currentDataFiles.Count
    dataBuildMetadataDiffs = @($dataDiffs)
    newDataBuildFiles = @($newDataPaths)
    sourceIntegrityPassed = ($sourceDiffs.Count -eq 0)
    dataBuildMetadataIntegrityPassed = ($dataDiffs.Count -eq 0 -and $newDataPaths.Count -eq 0)
    limitation = 'Large data/model/build binaries were checked by path, byte length and UTC modification time, not content hash. This check does not cover every file in .venv or untracked workspace files outside the original baseline.'
    expertNotesChecked = $noteRecords.Count
    expertFindingHeadings = $findingTotal
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evidenceRoot 'baseline-son-kontrol.json') -Encoding UTF8
$noteRecords | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $reviewRoot 'uzman-is-kayitlari.json') -Encoding UTF8
$record | Select-Object checkedUtc, sourceFilesChecked, dataBuildFilesChecked, currentDataBuildFileCount, sourceIntegrityPassed, dataBuildMetadataIntegrityPassed, expertNotesChecked, expertFindingHeadings | ConvertTo-Json
if (-not $record.sourceIntegrityPassed -or -not $record.dataBuildMetadataIntegrityPassed -or $noteRecords.Count -ne 10 -or $findingTotal -ne 53) { exit 1 }
