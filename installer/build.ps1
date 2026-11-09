<#
.SYNOPSIS
  Build the RelayVSR Studio GUI bundle (PyInstaller) and the Windows installer (Inno Setup).

.DESCRIPTION
  Steps:
    1. create a small build virtualenv with PySide6-Essentials + PyInstaller (no torch),
    2. build the relayvsr_studio wheel (installed into the managed engine env at first run),
    3. download uv.exe (environment manager) and vc_redist.x64.exe,
    4. run PyInstaller,
    5. run ISCC to produce build\installer\RelayVSR-Studio-Setup-<version>.exe

.PARAMETER WorkDir
  Scratch/output folder (default: <repo>\build). Use a short path on Windows.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File installer\build.ps1 -WorkDir E:\relayvsr-dev\build
#>
[CmdletBinding()]
param(
  [string]$WorkDir = "",
  [string]$Python = "python",
  [switch]$SkipInstaller,
  [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) { $WorkDir = Join-Path $Repo "build" }
$WorkDir = [System.IO.Path]::GetFullPath($WorkDir)
New-Item -ItemType Directory -Force -Path $WorkDir, "$WorkDir\wheels", "$WorkDir\tools", "$WorkDir\redist" | Out-Null

# ---- version from the package
$initPy = Get-Content (Join-Path $Repo "relayvsr_studio\__init__.py") -Raw
if ($initPy -match '__version__\s*=\s*"([^"]+)"') { $Version = $Matches[1] } else { throw "version not found" }
Write-Host "==> RelayVSR Studio $Version  (work dir: $WorkDir)"

# ---- build venv (GUI deps only)
$VenvDir = Join-Path $WorkDir "venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not $SkipVenv -or -not (Test-Path $VenvPy)) {
  Write-Host "==> Creating build environment"
  & $Python -m venv $VenvDir
  & $VenvPy -m pip install --upgrade pip wheel setuptools build | Out-Null
  & $VenvPy -m pip install "PySide6-Essentials>=6.7,<7" "pyinstaller>=6.10" | Out-Null
}

# ---- app wheel
Write-Host "==> Building wheel"
Get-ChildItem "$WorkDir\wheels" -Filter "relayvsr_studio-*.whl" -ErrorAction SilentlyContinue | Remove-Item -Force
Push-Location $Repo
try {
  & $VenvPy -m build --wheel --outdir "$WorkDir\wheels" --no-isolation
  if ($LASTEXITCODE -ne 0) { throw "wheel build failed" }
} finally { Pop-Location }

# ---- uv.exe
$UvExe = Join-Path $WorkDir "tools\uv.exe"
if (-not (Test-Path $UvExe)) {
  Write-Host "==> Downloading uv"
  $zip = Join-Path $WorkDir "tools\uv.zip"
  Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
  Expand-Archive -Path $zip -DestinationPath "$WorkDir\tools\uvtmp" -Force
  $found = Get-ChildItem "$WorkDir\tools\uvtmp" -Recurse -Filter "uv.exe" | Select-Object -First 1
  Copy-Item $found.FullName $UvExe -Force
  Remove-Item "$WorkDir\tools\uvtmp" -Recurse -Force
  Remove-Item $zip -Force
}

# ---- VC++ redistributable (installed only when missing on the target machine)
$Redist = Join-Path $WorkDir "redist\vc_redist.x64.exe"
if (-not (Test-Path $Redist)) {
  Write-Host "==> Downloading vc_redist.x64.exe"
  Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $Redist
}

# ---- Windows version resource
$parts = ($Version -split '[.-]')[0..2] | ForEach-Object { [int]$_ }
while ($parts.Count -lt 3) { $parts += 0 }
$verTuple = "($($parts[0]), $($parts[1]), $($parts[2]), 0)"
@"
VSVersionInfo(
  ffi=FixedFileInfo(filevers=$verTuple, prodvers=$verTuple, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('CompanyName', 'RelayVSR Studio contributors'),
    StringStruct('FileDescription', 'RelayVSR Studio - video upscaler'),
    StringStruct('FileVersion', '$Version'),
    StringStruct('InternalName', 'RelayVSR Studio'),
    StringStruct('LegalCopyright', 'Apache-2.0'),
    StringStruct('OriginalFilename', 'RelayVSR Studio.exe'),
    StringStruct('ProductName', 'RelayVSR Studio'),
    StringStruct('ProductVersion', '$Version')])]),
  VarFileInfo([VarStruct('Translation', [1033, 1200])])]
)
"@ | Set-Content -Encoding utf8 (Join-Path $WorkDir "version_info.txt")

# ---- PyInstaller
Write-Host "==> Running PyInstaller"
$env:RVS_BUILD_DIR = $WorkDir
$env:PYTHONDONTWRITEBYTECODE = "1"
& $VenvPy -m PyInstaller --noconfirm --clean --distpath "$WorkDir\dist" --workpath "$WorkDir\pyi" (Join-Path $Repo "installer\relayvsr_studio.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
$ExePath = Join-Path $WorkDir "dist\RelayVSR Studio\RelayVSR Studio.exe"
if (-not (Test-Path $ExePath)) { throw "bundle not found: $ExePath" }
$size = (Get-ChildItem "$WorkDir\dist\RelayVSR Studio" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("    bundle size: {0:N0} MB" -f $size)

if ($SkipInstaller) { Write-Host "==> Skipping installer"; exit 0 }

# ---- Inno Setup
$Iscc = $null
foreach ($candidate in @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
  if (Test-Path $candidate) { $Iscc = $candidate; break }
}
if (-not $Iscc) { $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue; if ($cmd) { $Iscc = $cmd.Source } }
if (-not $Iscc) { throw "ISCC.exe (Inno Setup 6) not found" }
Write-Host "==> Running Inno Setup ($Iscc)"
& $Iscc "/DAppVersion=$Version" "/DBuildDir=$WorkDir" (Join-Path $Repo "installer\relayvsr-studio.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
$Setup = Join-Path $WorkDir "installer\RelayVSR-Studio-Setup-$Version.exe"
Write-Host ("==> Installer: {0} ({1:N1} MB)" -f $Setup, ((Get-Item $Setup).Length / 1MB))
