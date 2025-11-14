# nnUNet Segmentation App

A GUI-based application for medical image segmentation using the nnUNet framework, supporting DICOM to NIfTI conversion, automatic segmentation with pre-trained models, and STL export for 3D printing or further analysis.
We developed it as an on-site tool that requires no technical background. It is tested on Windows, designed for a workstation setup for either coworkers or medical personnel who need models/masks for studies. Pre-trained models might be shared in the future. Places where best to add customizable changes to suit specific needs are given below. 
## Overview

This tool provides an end-to-end pipeline for medical image segmentation:
1. DICOM to NIfTI conversion
2. Optional processing ("splitting" into left/right, Z-axis cutting, mesh-repair)
3. nnUNet-based automatic segmentation
4. 3D model generation in STL format

The application is designed to be user-friendly with a GUI interface that supports drag-and-drop functionality, customizable segmentation parameters, and batch processing capabilities.

## Features

- **User-friendly GUI** with drag-and-drop support
- **DICOM filtering** by series description to select specific scans or with group labels (e.g. bone window if given in series description etc.)
- ** Tested for different models** for various body parts:
  - Ankle (healthy and post op)
  - Shoulder
  - Pelvis
  - Whole leg
- **Pre-processing options**:
  - Left/right splitting for bilateral structures
  - Z-axis cutting for region-of-interest focus
  - Multi-patient folder processing
- **Customizable smoothing parameters** for each anatomical structure
- **STL export** with adjustable smoothing parameters
- **Progress tracking** during processing

## Installation

### Prerequisites

- Python 3.12+
- PyTorch (2.5.1) (higher versions have safe loading (weights_only) which does not work with nnUNet last I checked
- CUDA (recommended for faster processing)
- nnUNetV2

### Required Python Packages

See requirements.txt

### nnUNet Installation

Follow the [official nnUNet installation guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

Make sure to set environment variables:
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```
Than download all the remaining parts from requirements.txt (pip install -r requirements.txt , easy support for uv might be added at some point as it is faster) 
## Usage
0. Add paths for label and id jsons which need structuring as in the example and set up venv, best is torch --> nnUNetv2 --> rest from requirements.txt (had problems with keeping all in requirements.txt)
1. Launch the application:
   ```bash
   python segment_app.py
   ```
   or use modified launch.ps1 script (add shortcut to desktop for coworkers who are not used to CLI) 

2. Configure input and output paths:
   - Set the input path (DICOM data folder)
   - Optionally set custom output paths for STL files and label maps

3. Select scan indicators or use "Don't filter" for processing all series

4. Choose dataset ID and configuration:
  - either via ID or name 

5. Select configuration and folds

6. Optional processing:
   - Split NIFTIs into left and right
   - Cut along Z direction (specify Z-range)
   - Process multiple patient subfolders
   - Mesh fix (fix + removal of small fragments still combined will be split in the next push) 

7. Click "Submit" to start processing

8. Advanced: Edit segment parameters for customized smoothing

## Customizing Segment Parameters

Click "Edit Segment Params" to customize parameters for each anatomical structure:
- `smoothing`: Labelmap smoothing factor (Gaussian smoothing: 0.2-2.0)
- `mesh_smoothing_method`: Method used for mesh smoothing ( 'taubin'): PyVista implementation as WindowedSync
- `mesh_smoothing_iterations`: Number of smoothing iterations (150)
- `mesh_smoothing_factor`: Strength of smoothing per iteration (pass-band: smaller value equals stronger smoothing, default 0.1 recommended)


## Output Files

The application generates:
- NIfTI files from DICOM data
- Segmentation label maps
- STL files for 3D printing
- A `decoder.json` file mapping the, for the segmentation, renamed files to the original ones.
- 
### Changes
Change the output name in the multiprocessed.py line 73 to the case you need, we had some irregular Patient_IDs before the last push, so used PatientName twice as we need the naming schema to be {identifier1}_{identifier2}_{series_description}, so change one back to PaientId etc.  
Also added some helper functions in utils, that are task task-specific in the prep for training with nnUNet. 

## Json examples 

ids_dict : {
    "111": {
        "body_part": "ankle,
        "Path_to_results": {
            "3d_fullres": "Dataset111_ankle/nnUNetTrainer__nnUNetPlans__3d_fullres"  #path in the results folder, add more configs if available
        },
        "configurations": [
           "3d_fullres"  
        ],
        "prefix": "",   
        "suffix": "_0000"   #no automatic renaming for different modalities included as we exclusively worked with CT images 
    }
, } <br>
labels_dict: {
    "111": {
        "1": "Fibula",
        "2": "Talus",
        "3": "Tibia"
    },} 
## Explanation for certain choices

The filtering via the Series Description is optional. It will just get Exceptions for files it cannot transform, which usually are exam summaries or dosis infos. If there are a lot of scans for a patient, it is still advised to use the one with the smallest slice thickness, as it will reduce steps in the stls. 
In "splitting" into left and right we don't separate the files into left and right, but mask the undesired side. We apply it to the labelmaps as the background value is 0 for every file. This avoids alignment issues in the stl and allows the user to use one-sided labelmaps for other applications if needed. 
I chose both labelmap smoothing as well as mesh smoothing. The first to remove minor artifacts from the segmentation, like a stray misclassed voxel. The latter is to remove the step artifacts that can show up in the conversion using taubin smoothing ensuring next to no volume change. I compared the resulting STLs to ones created from the segmentation maps using 3DSlicers built-in conversion and there is no meaningful difference (they might use a similar pipeline). Still be careful to not oversmooth, especially the Gaussian smoothing.
As there is no medical certification, this should be only used for research and not for clinical use. 

### Troubleshooting
There are still some edge cases where the filtering does not work. Cannot figure out where the problem lies, as it affects some of the harmless-looking ones. But Series Descriptions are usually a mess anyway, with varying conventions and no consistency. So do not filter if not needed. 
Also make sure to adapt the target_dir_name in Dicom_splitter of multiprocessed.py, some scans have horrendous ids(including timecodes etc., maybe clean those with the metadata editer).    



### Common Issues

1. **CUDA/GPU errors**: Make sure your CUDA installation matches your PyTorch version.
2. **Memory errors**: Try processing with smaller batch sizes or on CPU if GPU memory is limited, or change the number of workers for the parallel processing especially for the stl conversion. 
3. **Odd DICOM tags**: The majority of the conventions for the Series description were tested, but there might be an odd one out. Check the terminal after the conversion to see if the filenames were handled correctly.
4. **PyTorch version**: Make sure the right PyTorch version (2.5.1) is installed, as there are compatibility issues with nnUNet and the newer version. 
### Reporting Issues

Please open an issue on GitHub with:
- Detailed error message
- Operating system information
- Python version
- GPU model (if applicable)
- Sample data (if possible)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Acknowledgments

- The nnUNet framework developers for their excellent segmentation architecture
