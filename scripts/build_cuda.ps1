$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$vsDev = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $vsDev)) {
    throw "Visual Studio Build Tools was not found at $vsDev. Install the C++ build tools workload."
}
$cudaSource = Join-Path $repo 'cuda\attention_forward.cu'
$outputDir = Join-Path $repo 'build\cuda'
$output = Join-Path $outputDir 'attention_forward.exe'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$command = "call `"$vsDev`" -arch=x64 && cd /d `"$repo`" && nvcc -O3 -arch=sm_86 `"$cudaSource`" -o `"$output`""
& cmd.exe /s /c $command
if ($LASTEXITCODE -ne 0) { throw "CUDA compilation failed with exit code $LASTEXITCODE" }
Write-Host "Built $output"
