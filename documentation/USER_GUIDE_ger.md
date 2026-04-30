Diese Anwendung bietet eine End-to-End-Pipeline zur Konvertierung von medizinischen Rohdaten (DICOM) in 3D-Modelle (STL) und analysebereite Masken (NIfTI).

## Voraussetzungen
- **Daten**: Ein Ordner mit DICOM-Serien (Struktur egal) oder ein Ordner mit NIFTIs (flache Struktur, keine Unterordner).
- **Hardware**: Eine Workstation mit NVIDIA-GPU (empfohlen).
- **Python**: Stellen Sie sicher, dass die Umgebung aktiv ist. Starten Sie mit der `launch.ps1`, um Venv-Probleme zu vermeiden.

---

## Ordnerstruktur 

```text
root
    ├── input
        ├── dicom_folder_1
        │   └── dicom_scans 1
        └── dicom_folder_2
            └── dicom_folder_2.1
               └── dicoms_scans 2
    ├── sortiert
        ├── scan_1/
        └── scan_2/
    ├── NIFTI
        ├── scan_1.nii.gz
        └── scan_2.nii.gz
    ├── NIFTI_cut (only when cutting enabled)
        ├── scan_1_cut.nii.gz
        └── scan_2_cut.nii.gz
    ├── label
    ├── label_lowres (only for the lowres/cascade models)
    ├── stl
        ├── STL_Scan1
        │   ├── Scan1_Part1.stl
        │   └── Scan1_Part2.stl
        └── STL_Scan2
            ├── Scan2_Part1.stl
            └── Scan2_Part2.stl
    ├── logs
        └── .log
    ├── HU-Analytics
        └── hu_analysis.xlsx
    ├── decoder.json
    ├── stl_metadata.json
    ├── stl_checkpoint.json
    └── run_paramter.json
    
```
---
## 1. Pfade einrichten
Input Path: Ziehen Sie Ihren Ordner mit DICOM/NIFTI-Dateien per Drag & Drop in das Feld.

Output Paths: Legen Sie fest, wo STLs und NIfTIs gespeichert werden sollen. Falls leer, wird ein Unterordner im Quellverzeichnis erstellt.

## 1.1 NIFT Inpit / Umwandlungsskip 
Wenn direkt niftis genommen werden, den ordner mit den niftis ins input feld geben. erkennt diese automatisch und aktiviert den entsprecheden button. 
Bei rerun mit z.b. anderen paramatern oder nur HU kann der zweite button getoggelt werden. Dabei die normalen dicoms benutzen im input. 

## 2. Modell auswählen (Dataset ID)
Wählen Sie die passende Dataset ID (z. B. 111 für Knöchel) oder nutzen Sie das Dropdown-Menü für den Körperteil.

Wählen Sie die Configuration (meist 3d_fullres) und die Folds (Standard: alle 5 für maximale Genauigkeit).

## 3. Filterung & Vorverarbeitung (Optional)
Scan Indicators: Falls der Patientenordner viele Scans enthält (Loc-Scans, Dosisberichte), wählen Sie nur die hochauflösenden Serien (z. B. "0.8mm") oder nutzen Sie den Gruppenfilter (z. B. Knochenfenster "KF").

Split X-Axis: Aktivieren Sie dies für bilaterale Strukturen (z. B. linker/rechter Knöchel), um sie in separate Dateien zu trennen.

Crop Volume: Fokus auf eine bestimmte Region (ROI). Eingabe in RAS (Standard) oder LPS (üblich in Viewern wie MicroDicom) möglich auch prozentual, Stop zum check möglich. 



Glättungsparameter: Können separat für jeden Teil eingestellt werden. Manche Geometrien benötigen stärkere Glättung.

## 4. Pipeline starten
Klicken Sie auf Submit.

Verfolgen Sie den Fortschritt am Fortschrittsbalken und der Statuszeile unten.

Dashboard: Wenn Sie mehr als 10 Fälle verarbeiten, startet automatisch ein Qualitätskontroll-Dashboard, um Ausreißer-Segmentierungen mittels DBSCAN-Clustering zu identifizieren.

📧 Benachrichtigungen
Geben Sie Ihre E-Mail-Adresse an. Die App sendet eine Nachricht, sobald die Verarbeitung abgeschlossen ist oder falls ein Fehler auftritt.