# nnUNet Segmentation App

A GUI-based application for medical image segmentation using the nnUNet framework, supporting DICOM to NIfTI conversion, automatic segmentation with pre-trained models, and STL export for 3D printing or further analysis.

## Overview

This tool provides an end-to-end pipeline for medical image segmentation:
1. DICOM to NIfTI conversion
2. Optional processing ("splitting" into left/right, Z-axis cutting, mesh-repair)
3. nnUNet-based automatic segmentation
4. 3D model generation in STL format

The application is designed to be user-friendly with a GUI interface that supports drag-and-drop functionality, customizable segmentation parameters, and batch processing capabilities.

## Features

- **User-friendly GUI** with drag-and-drop support
- **DICOM filtering** by series description to select specific scans as everything above 1mm slice thickness 
- ** Tested for different models** for various body parts:
  - Ankle (healthy and post op)
  - Shoulder
  - Pelvis
- **Pre-processing options**:
  - Left/right splitting for bilateral structures
  - Z-axis cutting for region-of-interest focus
  - Multi-patient folder processing
- **Customizable smoothing parameters** for each anatomical structure
- **STL export** with adjustable smoothing parameters
- **Progress tracking** during processing

## Installation

### Prerequisites

- Python 3.6+
- PyTorch
- CUDA (recommended for faster processing)
- nnUNetV2

### Required Python Packages

```bash
pip install tkinter tkinterdnd2 numpy dicom2nifti pydicom torch nibabel scikit-image scipy pyvista numpy-stl
```

### nnUNet Installation

Follow the [official nnUNet installation guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

Make sure to set environment variables:
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Usage
0. Add paths for label and id jsons which need structuring as in the examaple
1. Launch the application:
   ```bash
   python segment_app.py
   ```

2. Configure input and output paths:
   - Set the input path (DICOM data folder)
   - Optionally set custom output paths for STL files and label maps

3. Select scan indicators or use "Don't filter" for processing all series

4. Choose dataset ID and configuration:
  - either via ID or name 

5. Select configuration and folds

6. Optional pre-processing:
   - Split NIFTIs into left and right
   - Cut along Z direction (specify Z-range)
   - Process multiple patient subfolders

7. Click "Submit" to start processing

8. Advanced: Edit segment parameters for customized smoothing

## Customizing Segment Parameters

Click "Edit Segment Params" to customize parameters for each anatomical structure:
- `smoothing`: Labelmap smoothing factor (Gaussian smoothing: 1.0-3.0)
- `mesh_smoothing_method`: Method used for mesh smoothing ( 'taubin'): PyVista implementation as WindowedSync
- `mesh_smoothing_iterations`: Number of smoothing iterations
- `mesh_smoothing_factor`: Strength of smoothing per iteration (pass-band: smaller value equals stronger smoothing, default 0.1 recommended)


## Output Files

The application generates:
- NIfTI files from DICOM data
- Segmentation label maps
- STL files for 3D printing
- A `decoder.json` file mapping the, for the segmentation, renamed files to the original ones.

## Json examples 

ids_dict = {
    "111": {
        "body_part": "ankle,
        "Path_to_results": {
            "3d_fullres": "Dataset111_ankle/nnUNetTrainer__nnUNetPlans__3d_fullres"  #patch in the results folder, add more config if available
        },
        "configurations": [
           "3d_fullres"  
        ],
        "prefix": "",   
        "suffix": "_0000"   #no autamtic renaming for different modalities included as we exclusively worked with CT images 
    }
, } <br / >
labels_dict {
    "111": {
        "1": "Fibula",
        "2": "Talus",
        "3": "Tibia"
    },} 
## Explanation for certain choices

The filtering via the Series Description is optional. It will just get Exceptions for files it cannot transform, which usually are exam summaries or dosis infos. If there are a lot of scans for a patient, it is still advised to use the one with the smallest slice thickness as it will reduce steps in the stls. 
For now, dictionaries are stored in the code as more models are added, moving them to an external json file and loading is advised. 
In "splitting" into left and right we don't seperate the files into left and right, but mask the undesired side. We apply it to the labelmaps as the background value is 0 for every file. This avoids alignment issues in the stl and allows the user to use one-sided labelmaps for other applications if needed. 
I chose both labelmap smoothing as well as mesh smoothing. The first to remove minor artifacts from the segmentation, like a stay misclassed voxel. The latter to remove the step artifacts that can show up in the conversion using taubin smoothing ensure next to no volume change. I compared the resulting STLs to ones created from the segmentation maps using 3DSlicers built in conversion and there is no meaningful difference. 
As there is no medical certification, this should be only used for research and not for clinical use. 

## Troubleshooting

### Common Issues

1. **CUDA/GPU errors**: Make sure your CUDA installation matches your PyTorch version.
2. **Memory errors**: Try processing with smaller batch sizes or on CPU if GPU memory is limited.
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
