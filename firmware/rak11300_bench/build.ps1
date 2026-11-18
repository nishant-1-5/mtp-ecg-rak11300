# Builds rak11300_bench.uf2 with the portable toolchain in <repo>/.toolchain
# (layout described in firmware/rak11300_bench/README.md).
#   powershell -ExecutionPolicy Bypass -File firmware/rak11300_bench/build.ps1
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$repo = Resolve-Path (Join-Path $here '..\..')
$tc = Join-Path $repo '.toolchain'

$env:PICO_SDK_PATH = Join-Path $tc 'pico-sdk'
$env:PICO_TOOLCHAIN_PATH = $tc
$cmakeDir = Get-ChildItem $tc -Directory -Filter 'cmake-*' | Select-Object -First 1
$env:PATH = (Join-Path $tc 'bin') + ';' + (Join-Path $cmakeDir.FullName 'bin') + ';' +
            (Join-Path $tc 'ninja') + ';' + $env:PATH

$build = Join-Path $here 'build'
$picotoolDir = Join-Path $tc 'picotool\picotool'
cmake -S $here -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release "-Dpicotool_DIR=$picotoolDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build $build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $tc 'bin\arm-none-eabi-size.exe') (Join-Path $build 'rak11300_bench.elf')
