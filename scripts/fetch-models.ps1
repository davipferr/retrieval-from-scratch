# Downloads the ONNX Runtime DLL and the bge-small-en-v1.5 embedding model into models/.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/fetch-models.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = Split-Path -Parent $PSScriptRoot
$models = Join-Path $root "models"
$ortVersion = "1.31.0"

# --- ONNX Runtime (only the DLLs are kept) ---
$ortDir = Join-Path $models "onnxruntime"
if (-not (Test-Path (Join-Path $ortDir "onnxruntime.dll"))) {
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ort-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $zip = Join-Path $tmp "ort.zip"
    $url = "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/onnxruntime-win-x64-$ortVersion.zip"
    Write-Host "downloading $url"
    curl.exe -fL --retry 3 -o $zip $url
    Expand-Archive $zip -DestinationPath $tmp
    New-Item -ItemType Directory -Force $ortDir | Out-Null
    Get-ChildItem $tmp -Recurse -Filter "onnxruntime*.dll" | Copy-Item -Destination $ortDir
    Remove-Item -Recurse -Force $tmp
}

# --- Embedding model ---
$modelDir = Join-Path $models "bge-small-en-v1.5"
New-Item -ItemType Directory -Force $modelDir | Out-Null
$base = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"
foreach ($file in @("onnx/model.onnx", "vocab.txt")) {
    $dest = Join-Path $modelDir (Split-Path -Leaf $file)
    if (-not (Test-Path $dest)) {
        Write-Host "downloading $base/$file"
        curl.exe -fL --retry 3 -o $dest "$base/$file"
    }
}
Get-ChildItem $ortDir, $modelDir | Select-Object Name, Length | Format-Table
