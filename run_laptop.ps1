param (
    [Parameter(Mandatory=$false)]
    [string]$Source = "rtsp",

    [Parameter(Mandatory=$false)]
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

# Locate python executable in virtual environment or fallback to system python
$pythonExec = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $pythonExec = ".\venv\Scripts\python.exe"
}

Write-Host "[+] Activating Gun Detection Pipeline..." -ForegroundColor Green
Write-Host "[+] Source: $Source" -ForegroundColor Cyan

if ($Debug) {
    Write-Host "[+] Debug mode enabled." -ForegroundColor Yellow
}

# Run main.py with the provided source argument
& $pythonExec main.py --source $Source
