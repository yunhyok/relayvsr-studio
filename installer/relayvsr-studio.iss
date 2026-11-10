; Inno Setup script for RelayVSR Studio.
; Build with: ISCC.exe /DAppVersion=0.2.0 /DBuildDir=<build dir> relayvsr-studio.iss

#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif
#ifndef BuildDir
  #define BuildDir "..\build"
#endif
#define AppName "RelayVSR Studio"
#define AppPublisher "RelayVSR Studio contributors"
#define AppURL "https://github.com/yunhyok/relayvsr-studio"
#define AppExeName "RelayVSR Studio.exe"

[Setup]
AppId={{7E4C1D1A-6B4E-4D1B-9C7A-2F3B0F5E9A21}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={userpf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#BuildDir}\installer
OutputBaseFilename=RelayVSR-Studio-Setup-{#AppVersion}
SetupIconFile=..\relayvsr_studio\app\resources\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
LicenseFile=..\LICENSE
ShowLanguageDialog=auto
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#BuildDir}\dist\{#AppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildDir}\redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: VCRedistNeeded

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Microsoft Visual C++ Runtime..."; Check: VCRedistNeeded; Flags: waituntilterminated
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Messages]
english.FinishedLabel=Setup has finished installing [name]. On first launch the application downloads its inference engine (PyTorch CUDA 12.8, about 3.5 GB) and the RelayVSR model weights (about 10 GB per model) into %LOCALAPPDATA%\RelayVSRStudio. An NVIDIA GPU with a recent driver is required.
korean.FinishedLabel=[name] 설치가 끝났습니다. 처음 실행하면 추론 엔진(PyTorch CUDA 12.8, 약 3.5 GB)과 RelayVSR 모델 가중치(모델당 약 10 GB)를 %LOCALAPPDATA%\RelayVSRStudio 에 내려받습니다. 최신 드라이버가 설치된 NVIDIA GPU가 필요합니다.

[Code]
function VCRedistNeeded: Boolean;
var
  Installed: Cardinal;
begin
  Result := True;
  if RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) then
    Result := Installed <> 1
  else if RegQueryDWordValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) then
    Result := Installed <> 1;
end;
