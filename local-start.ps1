$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not $env:COMFY_ROOT) {
  $candidates = @(
    (Resolve-Path "$RepoRoot\..\ComfyUI" -ErrorAction SilentlyContinue),
    "C:\ComfyUI_windows_portable\ComfyUI"
  ) | Where-Object { $_ }

  foreach ($candidate in $candidates) {
    $path = [string]$candidate
    if ((Test-Path (Join-Path $path "main.py")) -or (Test-Path (Join-Path $path "models"))) {
      $env:COMFY_ROOT = $path
      break
    }
  }
}

if (-not $env:COMFY_ROOT) {
  throw "Could not detect COMFY_ROOT. Set `$env:COMFY_ROOT first."
}

if (-not $env:COMFY_URL) { $env:COMFY_URL = "http://127.0.0.1:8188" }
if (-not $env:UI_PORT) { $env:UI_PORT = "3000" }
if (-not $env:FAVORITES_PATH) { $env:FAVORITES_PATH = (Join-Path $RepoRoot "muse_worker\runtime\favorites.json") }

$pythonCandidates = @(
  "C:\ComfyUI_windows_portable\python_embeded\python.exe",
  "python"
)

$python = $null
foreach ($candidate in $pythonCandidates) {
  if ($candidate -eq "python") {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source; break }
  } elseif (Test-Path $candidate) {
    $python = $candidate
    break
  }
}

if (-not $python) {
  throw "Could not find Python. Install Python or use ComfyUI portable python."
}

Write-Host "Starting Comfy Vast Studio (docker3000_script)"
Write-Host "COMFY_ROOT=$env:COMFY_ROOT"
Write-Host "COMFY_URL=$env:COMFY_URL"
& $python -m pip install -r requirements.txt
& $python app.py
