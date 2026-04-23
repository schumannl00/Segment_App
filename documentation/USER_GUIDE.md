# User Guide: nnUNet Segmentation App

This application provides an end-to-end pipeline to convert raw medical imaging (DICOM) into 3D models (STL) and analysis-ready masks (NIfTI).

##  Prerequisites
- **Data**: A folder containing DICOM series (does not matter how it is structured) or a folder of NIFTIs (flat structure no subfolder)
- **Hardware**: A workstation with an NVIDIA GPU (recommended).
- **Python**: Ensure your environment is active and has all its dependencies installed. 
         Launch with launch.ps1, this removes any hurdles with venv. 

---

## Folder strcuture 

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

### 1. Set Up Your Paths
- **Input Path**: Drag and drop your folder containing DICOM/NIFTI files into the "Input Path" field.
- **Output Paths**: Set your desired locations for STL files (3D models) and NIfTI labelmaps. If left blank, these default to a subfolder in your input paths parent directory.

### 2. Select Your Model (Dataset ID)
- Choose the appropriate **Dataset ID** (e.g., `111` for Ankle) or use the **Dataset Name** dropdown to select by body part.
- Choose the **Configuration** (usually `3d_fullres`) and select the **Folds** (default is all 5 for maximum accuracy).

### 3. Filtering & Preprocessing (Optional)
- **Scan Indicators**: If your patient folder has multiple scans (Localizers, Dose Reports, Contrast), select only the high-resolution series (e.g., "0.8mm") or use the group filter to filter out specific groups like bone window (KF)
- **Split X-Axis**: Check this if you are processing bilateral structures (like left/right ankles) and want them separated into different files. 
- **Crop Volume**: Use this to focus on a specific Region of Interest (ROI). You can input coordinates in **RAS** (Standard) or **LPS** (common in viewers like MicroDicom). Also percentage based input is possible, which is often fine enough. 

- **Edit Smoothing parameters**: Smoothing parameters can be set seperate for each part, as some might need stronger smoothing. It should not cause shifts, but odd geometries might cause some problems. 
 
### 4. Running the Pipeline
- Click **Submit**. 
- Monitor the **Progress Bar** and the **Status Bar** at the bottom.
- **Dashboard**: If you process more than 10 cases, a quality control dashboard will automatically launch to help identify outlier segmentations using DBSCAN clustering.

---

## 📧 Notifications
Enter your email. The app will send a notification once the processing is complete or if an error occurs.