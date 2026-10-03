<#
.SYNOPSIS
  Configure and build everything (Release) with MSVC 14.4x (VS 2022) + CUDA + Ninja.
  CUDA 13.1 rejects MSVC 14.50 (VS 2026), so the VS 2022 environment is imported explicitly.
#>
param([string]$BuildDir = "$PSScriptRoot\build", [string]$Config = "Release")
$ErrorActionPreference = 'Stop'
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    if (Test-Path "$PSScriptRoot\.tools\ninja\ninja.exe") { $env:Path = "$PSScriptRoot\.tools\ninja;$env:Path" }
    else { throw "ninja not found: winget install Ninja-build.Ninja" }
}
cmake -S $PSScriptRoot -B $BuildDir -G Ninja "-DCMAKE_BUILD_TYPE=$Config"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build $BuildDir
exit $LASTEXITCODE
