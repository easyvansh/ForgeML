$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$vsDev='C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat'
$out=Join-Path $repo 'build\cuda\attention_backward.exe'
New-Item -ItemType Directory -Force (Split-Path $out)|Out-Null
$cmd="call `"$vsDev`" -arch=x64 && cd /d `"$repo`" && nvcc -O3 -arch=sm_86 cuda\attention_backward.cu -o `"$out`""
& cmd.exe /s /c $cmd
if($LASTEXITCODE){throw "backward CUDA compilation failed: $LASTEXITCODE"}
Write-Host "Built $out"
