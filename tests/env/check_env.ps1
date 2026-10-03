<#
.SYNOPSIS
  Phase 0 environment check. Runs every verification command from PLAN.md and
  prints a PASS/MISSING/FAIL summary.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tests\env\check_env.ps1
  powershell -ExecutionPolicy Bypass -File tests\env\check_env.ps1 -JavaHome C:\path\to\jdk-25 -NinjaDir C:\path\to\ninja
#>
param(
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$NinjaDir = "",
    [string]$BuildDir = "$env:TEMP\mccuda-envcheck"
)

$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$results = [System.Collections.Generic.List[object]]::new()

function Report($name, $status, $detail) {
    $results.Add([pscustomobject]@{ Check = $name; Status = $status; Detail = $detail })
}

function Section($title) { Write-Host ""; Write-Host "===== $title" -ForegroundColor Cyan }

# CUDA 13.1 accepts MSVC 19.2x-19.4x only, so import the VS 2022 x64 environment
# explicitly (VS 2026 / MSVC 14.50 is also installed on this machine and is rejected).
Section "MSVC (VS 2022 x64 environment)"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (Test-Path $vcvars) {
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
    $clBanner = (cmd /c "cl 2>&1" | Select-Object -First 1)
    Write-Host $clBanner
    if ($clBanner -match 'Version 19\.(\d+)') {
        if ([int]$matches[1] -lt 50) { Report 'MSVC' 'PASS' $clBanner } else { Report 'MSVC' 'FAIL' "$clBanner (CUDA 13.1 needs < 19.50)" }
    } else { Report 'MSVC' 'FAIL' 'cl did not run' }
} else { Report 'MSVC' 'MISSING' "vcvars64.bat not found at $vcvars" }

if ($NinjaDir) { $env:Path = "$NinjaDir;$env:Path" }
if ($JavaHome) { $env:JAVA_HOME = $JavaHome; $env:Path = "$JavaHome\bin;$env:Path" }

Section "NVIDIA driver / GPU"
$gpu = nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader 2>&1
Write-Host $gpu
if ($LASTEXITCODE -eq 0 -and $gpu -match '12\.0') { Report 'Driver/GPU' 'PASS' $gpu } else { Report 'Driver/GPU' 'FAIL' "$gpu" }

Section "CUDA toolkit"
$nv = nvcc --version 2>&1; $nv | Write-Host
$rel = ($nv | Select-String 'release').ToString()
if ($LASTEXITCODE -eq 0) { Report 'CUDA nvcc' 'PASS' $rel.Trim() } else { Report 'CUDA nvcc' 'MISSING' 'nvcc not on PATH' }

Section "CMake / Ninja"
$cmOut = cmake --version 2>&1; $cmOk = ($LASTEXITCODE -eq 0)
$cm = $cmOut | Select-Object -First 1; Write-Host $cm
if ($cmOk) { Report 'CMake' 'PASS' $cm } else { Report 'CMake' 'MISSING' '' }
$nj = Get-Command ninja -ErrorAction SilentlyContinue
if ($nj) { $nver = ninja --version; Write-Host "ninja $nver ($($nj.Source))"; Report 'Ninja' 'PASS' "ninja $nver" }
else { Write-Host "ninja not found"; Report 'Ninja' 'MISSING' 'winget install Ninja-build.Ninja' }

Section "CUDA hello world for sm_120 (nvcc direct)"
New-Item -ItemType Directory -Force $BuildDir | Out-Null
$exe = Join-Path $BuildDir 'hello_sm120.exe'
nvcc -O3 -arch=sm_120 -fmad=false -o $exe (Join-Path $here 'hello_sm120.cu') 2>&1 | Where-Object { $_ -notmatch 'hello_sm120.cu$|cudafe1.cpp$' } | Write-Host
if (Test-Path $exe) {
    cuobjdump --list-elf $exe | Write-Host
    & $exe | Write-Host
    if ($LASTEXITCODE -eq 0) { Report 'CUDA sm_120 run' 'PASS' 'hello + FMA/subnormal parity checks passed' } else { Report 'CUDA sm_120 run' 'FAIL' "exit $LASTEXITCODE" }
} else { Report 'CUDA sm_120 run' 'FAIL' 'compile failed' }

Section "CMake + Ninja + CUDA configure/build"
if ($nj) {
    $cb = Join-Path $BuildDir 'cmake'
    cmake -S $here -B $cb -G Ninja -DCMAKE_BUILD_TYPE=Release 2>&1 | Select-String -Pattern 'compiler identification|Build files|Error' | Write-Host
    cmake --build $cb 2>&1 | Select-Object -Last 2 | Write-Host
    if ($LASTEXITCODE -eq 0) { Report 'CMake CUDA build' 'PASS' 'Ninja generator, sm_120, -fmad=false' } else { Report 'CMake CUDA build' 'FAIL' 'see output' }
} else { Report 'CMake CUDA build' 'SKIPPED' 'needs Ninja (no CUDA MSBuild integration in VS either)' }

Section "Nsight Systems"
$nsysExe = (Get-Command nsys -ErrorAction SilentlyContinue).Source
if (-not $nsysExe) { $nsysExe = Get-ChildItem 'C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe' -ErrorAction SilentlyContinue | Select-Object -Last 1 -ExpandProperty FullName }
if ($nsysExe) {
    $nsv = & $nsysExe --version; Write-Host $nsv
    & $nsysExe profile --force-overwrite true -o (Join-Path $BuildDir 'probe') --trace=cuda $exe *> $null
    if ($LASTEXITCODE -eq 0) { Report 'Nsight Systems' 'PASS' "$nsv (not on PATH: $nsysExe)" } else { Report 'Nsight Systems' 'FAIL' 'profile run failed' }
} else { Report 'Nsight Systems' 'MISSING' '' }

Section "Nsight Compute"
$ncv = (ncu --version 2>&1 | Select-Object -Last 1); Write-Host $ncv
$ncuOut = ncu --kernel-name contraction --section SpeedOfLight $exe 2>&1 | Out-String
if ($ncuOut -match 'ERR_NVGPUCTRPERM') { Write-Host 'ERR_NVGPUCTRPERM'; Report 'Nsight Compute' 'FAIL' "$ncv; counters are admin-only (ERR_NVGPUCTRPERM)" }
elseif ($ncuOut -match 'Duration') { Report 'Nsight Compute' 'PASS' $ncv }
else { Report 'Nsight Compute' 'FAIL' 'unexpected output' }

Section "Java / Gradle / vanilla oracle smoke test"
$jv = cmd /c "java -version 2>&1" | Select-Object -First 1
Write-Host "java: $jv"
if ($jv -match 'version "25') { Report 'JDK 25' 'PASS' $jv } else { Report 'JDK 25' 'MISSING' "found: $jv" }
if ($jv -match 'version "25') {
    Push-Location (Join-Path $here 'oracle-smoke')
    $gv = (.\gradlew.bat --version 2>$null | Select-String '^Gradle ').ToString(); Write-Host $gv
    $smoke = .\gradlew.bat --console=plain -q runSmoke 2>$null | Out-String
    Pop-Location
    Write-Host $smoke
    if ($smoke -match 'SMOKE: PASS') { Report 'Oracle smoke (MC 26.3)' 'PASS' "$gv, Loom 1.18.2" } else { Report 'Oracle smoke (MC 26.3)' 'FAIL' 'see output' }
} else { Report 'Oracle smoke (MC 26.3)' 'SKIPPED' 'needs JDK 25' }

Section "Python / git"
$py = python --version 2>&1; Write-Host $py
$libs = python -c "import numpy, matplotlib; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)" 2>&1; Write-Host $libs
if ($LASTEXITCODE -eq 0) { Report 'Python+numpy+matplotlib' 'PASS' "$py; $libs" } else { Report 'Python+numpy+matplotlib' 'FAIL' "$libs" }
$gt = git --version; Write-Host $gt
if ($LASTEXITCODE -eq 0) { Report 'git' 'PASS' $gt } else { Report 'git' 'MISSING' '' }

Section "Summary"
$results | Format-Table -AutoSize -Wrap | Out-String -Width 200 | Write-Host
if ($results | Where-Object { $_.Status -ne 'PASS' }) { exit 1 } else { exit 0 }
