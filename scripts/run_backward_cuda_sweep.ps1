$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\cuda\attention_backward.exe'
if (-not (Test-Path -LiteralPath $exe)) { & (Join-Path $PSScriptRoot 'build_backward_cuda.ps1') }
$output = Join-Path $repo 'results\cuda_attention_backward.jsonl'
Push-Location $repo
try {
  $lines = foreach ($context in 127,256,513) { & $exe $context }
  $lines | Set-Content -Encoding utf8 $output
  Write-Host "Saved $output"
  $lines
} finally { Pop-Location }
