$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\cuda\attention_tiled.exe'
if (-not (Test-Path -LiteralPath $exe)) { & (Join-Path $PSScriptRoot 'build_tiled_cuda.ps1') }
$output = Join-Path $repo 'results\cuda_attention_tiled.jsonl'
Push-Location $repo
try {
  $lines = foreach ($context in 128,256,512) { & $exe $context }
  $lines | Set-Content -Encoding utf8 $output
  Write-Host "Saved $output"
  $lines
} finally { Pop-Location }
