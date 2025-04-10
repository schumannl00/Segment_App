# nnUNet Segmentation App

A GUI-based application for medical image segmentation using the nnUNet framework, supporting DICOM to NIfTI conversion, automatic segmentation with pre-trained models, and STL export for 3D printing.

## Overview

This tool provides an end-to-end pipeline for medical image segmentation:
1. DICOM to NIfTI conversion
2. Optional pre-processing (splitting into left/right, Z-axis cutting)
3. nnUNet-based automatic segmentation
4. 3D model generation in STL format

The application is designed to be user-friendly with a GUI interface supporting drag-and-drop functionality, customizable segmentation parameters, and batch processing capabilities.

## Features

- **User-friendly GUI** with drag-and-drop support
- **DICOM filtering** by series description
- **Pre-trained models** for various body parts:
  - Ankle (healthy and mixed pathology)
  - Shoulder
  - Pelvis
- **Pre-processing options**:
  - Left/right splitting for bilateral structures
  - Z-axis cutting for region-of-interest focus
  - Multi-patient folder processing
- **Customizable segmentation parameters** for each anatomical structure
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

1. Launch the application:
   ```bash
   python segment_app.py
   ```

2. Configure input and output paths:
   - Set the input path (DICOM data folder)
   - Optionally set custom output paths for STL files and label maps

3. Select scan indicators or use "Don't filter" for processing all series

4. Choose dataset ID and configuration:
   - ID 111: Healthy ankle
   - ID 112: Mixed pathology ankle
   - ID 114: Shoulder
   - ID 116: Pelvis

5. Select configuration and folds

6. Optional pre-processing:
   - Split NIFTIs into left and right
   - Cut along Z direction (specify Z-range)
   - Process multiple patient subfolders

7. Click "Submit" to start processing

8. Advanced: Edit segment parameters for customized smoothing

## Customizing Segment Parameters

Click "Edit Segment Params" to customize parameters for each anatomical structure:
- `smoothing`: Overall smoothing factor (0.0-1.0)
- `mesh_smoothing_method`: Method used for mesh smoothing (e.g., 'taubin')
- `mesh_smoothing_iterations`: Number of smoothing iterations
- `mesh_smoothing_factor`: Strength of smoothing per iteration

## Pre-trained Models

The tool comes with several pre-trained models:

| ID  | Body Part          | Configuration | Description                  |
|-----|-------------------|---------------|------------------------------|
| 111 | Ankle (healthy)   | 3d_fullres    | Healthy ankle segmentation   |
| 112 | Ankle (mixed)     | 3d_fullres    | Mixed pathology ankle        |
| 114 | Shoulder          | 3d_fullres    | Shoulder segmentation        |
| 116 | Pelvis            | 3d_fullres    | Pelvis segmentation          |

## Output Files

The application generates:
- NIfTI files from DICOM data
- Segmentation label maps
- STL files for 3D printing
- A `decoder.json` file mapping file names to original identifiers

## Troubleshooting

### Common Issues

1. **CUDA/GPU errors**: Make sure your CUDA installation matches your PyTorch version.
2. **Memory errors**: Try processing with smaller batch sizes or on CPU if GPU memory is limited.
3. **Missing DICOM tags**: Ensure your DICOM files contain standard tags.
4. **Z-range issues**: When using Z-cuts, ensure values are within the image dimensions.

### Reporting Issues

Please open an issue on GitHub with:
- Detailed error message
- Operating system information
- Python version
- GPU model (if applicable)
- Sample data (if possible)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Insert your license information here]

## Acknowledgments

- The nnUNet framework developers for their excellent segmentation architecture
- Contributors to the various Python packages used in this project
