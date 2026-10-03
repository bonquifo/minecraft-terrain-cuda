<#
.SYNOPSIS
  Regenerates all reference data from the local Minecraft 26.3 jar (via Loom):
  the IR (data/gamedata/26.3) and the oracle dumps (data/oracle/26.3).
  Nothing produced here is committed (data/ is gitignored).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File oracle\regen.ps1
  powershell -ExecutionPolicy Bypass -File oracle\regen.ps1 -Command dump-chunks
#>
param([string]$Command = "all", [string]$JavaHome = "")
$ErrorActionPreference = 'Stop'
if ($JavaHome) { $env:JAVA_HOME = $JavaHome }
elseif (-not $env:JAVA_HOME -and (Test-Path "$PSScriptRoot\..\.tools\jdk-25")) { $env:JAVA_HOME = (Resolve-Path "$PSScriptRoot\..\.tools\jdk-25").Path }
if (-not $env:JAVA_HOME) { throw "JAVA_HOME must point to a JDK 25 (winget install EclipseAdoptium.Temurin.25.JDK)" }
Push-Location $PSScriptRoot
try {
    # The JDK prints deprecation warnings on stderr; with 'Stop', Windows PowerShell would turn
    # them into terminating errors when output is redirected. Success is the exit code.
    $ErrorActionPreference = 'Continue'
    & .\gradlew.bat --console=plain -q oracle "-Pcmd=$Command" 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "oracle failed with exit code $code" }
} finally {
    Pop-Location
}
