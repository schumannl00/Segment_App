# Get the folder where this script lives
$ScriptDir = $PSScriptRoot

# Define the python path inside the venv folder
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"

# Check if venv exists
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Could not find the python virtual environment at $PythonExe" -ForegroundColor Red
    Pause
    Exit
}

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "            nnU-Net Evaluation Script" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Prompt the user for input
$GT = Read-Host "Enter the Ground Truth folder path"
$Pred = Read-Host "Enter the Predictions folder path"
$StartLab = Read-Host "Enter the starting label (e.g., 1)"
$EndLab = Read-Host "Enter the ending label (e.g., 4)"
$Out = Read-Host "Enter output filename (Press Enter for 'summary.json' in Prediction folder)"

if ([string]::IsNullOrWhiteSpace($Out)) { $Out = "summary.json" }

Write-Host "`nRunning evaluation... Please wait.`n" -ForegroundColor Yellow

# Execute python script using the specific venv python.exe
& $PythonExe "$ScriptDir\eval.py" -gt $GT -pred $Pred -start $StartLab -end $EndLab -o $Out

Write-Host "`nDone!" -ForegroundColor Green
Pause