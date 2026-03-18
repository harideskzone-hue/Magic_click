; ══════════════════════════════════════════════════════════════════════════════
; Magic Click — Windows Inno Setup Script
; ══════════════════════════════════════════════════════════════════════════════
; Build with: iscc MagicClick_Setup.iss
; Requires: Inno Setup 6+ from https://jrsoftware.org/isdl.php
;
; Delivers:
;   - OS + disk space pre-checks
;   - Silent Python 3.11 install if Python not found or too old
;   - Runs installer_bootstrap.py (shows premium GUI progress)
;   - Desktop + Start Menu shortcuts
;   - Clean uninstaller
; ══════════════════════════════════════════════════════════════════════════════

#define MyAppName       "Magic Click"
#define MyAppVersion    "2.0.0"
#define MyAppPublisher  "Magic Click Team"
#define MyAppURL        "http://localhost:5001"
#define MyAppExeName    "MagicClick.bat"
#define InstallDir      "{autopf}\MagicClick"

[Setup]
AppId                 = {{7F3B2A1C-4D8E-4F9A-B2C3-1A2B3C4D5E6F}
AppName               = {#MyAppName}
AppVersion            = {#MyAppVersion}
AppPublisher          = {#MyAppPublisher}
AppPublisherURL       = {#MyAppURL}
DefaultDirName        = {#InstallDir}
DefaultGroupName      = {#MyAppName}
AllowNoIcons          = yes
OutputDir             = dist
OutputBaseFilename    = MagicClick_Setup
Compression           = lzma2/max
SolidCompression      = yes
WizardStyle           = modern
PrivilegesRequired    = admin
ArchitecturesAllowed  = x64
MinVersion            = 10.0
DiskSpaceMBRequired   = 3072
UninstallDisplayIcon  = {app}\installers\windows\icon.ico
SetupIconFile         = installers\windows\icon.ico
; Wizard pages
WizardImageFile       = installers\windows\wizard_banner.bmp
WizardSmallImageFile  = installers\windows\wizard_small.bmp

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=Magic Click will be installed on your computer.%n%n%nIMPORTANT — Before continuing:%n%n  • Connect to Wi-Fi or Ethernet (required for AI model download ~500 MB)%n  • Ensure you have at least 3 GB free disk space%n  • The first-run setup takes 5–15 minutes%n%nClick Next to continue.
FinishedLabel=Magic Click has been installed.%n%nDouble-click the desktop shortcut to launch the pipeline.%n%nThe first launch completes the AI model setup automatically.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "startmenuicon"; Description: "Create Start Menu shortcut"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Include entire project (excluding venv, captured videos, git)
Source: "..\..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; \
  Excludes: ".venv\*,__pycache__\*,*.pyc,.git\*,captured_videos\*,captured_shots\*,data\db\*"

[Icons]
Name: "{group}\{#MyAppName}";          Filename: "{app}\launchers\windows\MagicClick.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}";  Filename: "{app}\launchers\windows\MagicClick.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Step 1: Ensure Python 3.10+ is installed
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -File ""{app}\installers\windows\scripts\check_python.ps1"""; \
  Flags: waituntilterminated runhidden; \
  StatusMsg: "Checking Python installation..."

; Step 2: Run the premium installer bootstrap GUI
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -WindowStyle Hidden -Command ""$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User'); python '{app}\installers\installer_bootstrap.py'"""; \
  WorkingDir: "{app}"; \
  Flags: waituntilterminated; \
  StatusMsg: "Setting up Magic Click (this takes a few minutes)..."

; Step 3: Open dashboard after install
Filename: "{app}\launchers\windows\MagicClick.bat"; \
  WorkingDir: "{app}"; \
  Flags: postinstall nowait skipifsilent; \
  Description: "Launch Magic Click now"

[UninstallRun]
Filename: "cmd.exe"; Parameters: "/C rmdir /S /Q ""{app}\.venv"""; Flags: waituntilterminated runhidden

[Code]
// ── Pre-install: Windows 10+ check ───────────────────────────────────────────
function InitializeSetup(): Boolean;
var
  DiskFreeGB: Extended;
begin
  Result := True;

  // Windows 10 build 1903+ required
  if not (GetWindowsVersion >= $0A000000) then begin
    MsgBox('Magic Click requires Windows 10 or newer.' + #13#10 +
           'Please upgrade your operating system.', mbCriticalError, MB_OK);
    Result := False;
    Exit;
  end;

  // Disk space: GetSpaceOnDisk returns KB, convert to GB
  DiskFreeGB := GetSpaceOnDisk(ExpandConstant('{autopf}'), True) / (1024 * 1024);
  if DiskFreeGB < 3.0 then begin
    MsgBox('Not enough disk space.' + #13#10 +
           'Magic Click requires at least 3 GB of free space.' + #13#10 +
           Format('You have %.1f GB available.', [DiskFreeGB]), mbCriticalError, MB_OK);
    Result := False;
    Exit;
  end;
end;

// ── Show Windows Defender notice before install starts ────────────────────────
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if MsgBox(
    'Windows Security Notice' + #13#10#13#10 +
    'Windows Defender SmartScreen may show a warning because Magic Click ' +
    'is not yet code-signed.' + #13#10#13#10 +
    'This is normal for independently distributed software. The application ' +
    'is safe to run.' + #13#10#13#10 +
    'If SmartScreen blocks the installer:' + #13#10 +
    '  1. Click "More info"' + #13#10 +
    '  2. Click "Run anyway"' + #13#10#13#10 +
    'Do you want to continue with the installation?',
    mbConfirmation, MB_YESNO) = IDNO
  then
    Result := 'Installation cancelled by user.';
end;
