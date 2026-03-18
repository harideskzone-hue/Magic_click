# check_python.ps1 — Silently install Python 3.11 if not found or too old
# Run by Inno Setup during [Run] section with -ExecutionPolicy Bypass

$MinMajor = 3
$MinMinor = 10
$PythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$PythonInstaller = "$env:TEMP\python_installer.exe"
$LogFile = "$env:TEMP\magic_click_python_install.log"

function Write-Log { param($Msg) Add-Content $LogFile "[$(Get-Date -f 'HH:mm:ss')] $Msg" }

Write-Log "=== Python Check ==="

# Check current Python
function Find-Python {
    foreach ($cmd in @("python", "python3")) {
        $CmdPath = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($CmdPath) { return $CmdPath.Path }
    }
    
    # Check standard install paths
    $Paths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe",
        "C:\Program Files\Python3*\python.exe",
        "C:\Program Files (x86)\Python3*\python.exe"
    )
    foreach ($pattern in $Paths) {
        $found = Get-Item $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

$PyPath = Find-Python
$NeedInstall = $true

if ($PyPath) {
    $VerStr = & $PyPath --version 2>&1
    if ($VerStr -match 'Python (\d+)\.(\d+)') {
        $Maj = [int]$Matches[1]; $Min = [int]$Matches[2]
        Write-Log "Found Python $Maj.$Min"
        if ($Maj -gt $MinMajor -or ($Maj -eq $MinMajor -and $Min -ge $MinMinor)) {
            Write-Log "Python version OK"
            $NeedInstall = $false
        } else {
            Write-Log "Python $Maj.$Min is too old — installing 3.11"
        }
    }
}

if ($NeedInstall) {
    Write-Log "Downloading Python 3.11 from $PythonUrl"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonInstaller -UseBasicParsing
        Write-Log "Download complete. Running silent installer…"
        $Args = "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_pip=1"
        Start-Process -FilePath $PythonInstaller -ArgumentList $Args -Wait
        Write-Log "Python 3.11 installed successfully"
        Remove-Item $PythonInstaller -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Log "ERROR: $_"
        [System.Windows.Forms.MessageBox]::Show(
            "Failed to download Python automatically.`n`nPlease install Python 3.10+ manually from:`nhttps://www.python.org/downloads/`n`nThen run the Magic Click installer again.",
            "Python Installation Failed", "OK", "Error")
        exit 1
    }
}

Write-Log "Python check complete"
exit 0
