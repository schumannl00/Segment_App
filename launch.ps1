Set-Location -Path $PSScriptRoot

$ScriptPath = $PSScriptRoot 

$Env:nnUNet_raw = 'D:\nnUNet_raw'
$Env:nnUNet_preprocessed = 'D:\nnUNet_preprocessed'
$Env:nnUNet_results = 'D:\nnUNet_results'


$PythonExecutable = Join-Path -Path $ScriptPath -ChildPath ".venv\Scripts\python.exe"
$AppScript = Join-Path -Path $ScriptPath -ChildPath "Segmentierung_App.py" 

& $PythonExecutable $AppScript