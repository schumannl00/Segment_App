
# Core concepts 

This document provides a deep dive into the logic governing the nnUNet Segmentation App, intended for developers or researchers maintaining the system.

## 1. Data Ingestion & Standardization
The pipeline begins with raw DICOM data, which is notoriously inconsistent in meta data convention. So a lot of regex handling is needed to use any of that for file names. 
Ended up with quite a verbose setup using PatientID, PatientName, SeriesNumber, SeriesDescription and StudyDescription. This should cover all the weird cases. we get quite long names this way though as we do: 
```
pid = getattr(dcm, "PatientID", "UnknownID")  
pname = getattr(dcm, "PatientName", "UnknownName")
sdesc = getattr(dcm, "SeriesDescription", "UnknownSeries")
snum = getattr(dcm, "SeriesNumber", 0)
stdesc = getattr(dcm, "StudyDescription", "unknownStudy")
#would be nice if the radiology would have a unified format for them, but they can be anything 
folder = sort_dir / f"{pid}_{pname}_Series{snum}@{stdesc}_{sdesc}"  
``` 


DICOM Sorting (DICOM_splitter): Instead of moving files, the app creates Hard Links in the sortiert/ directory. This allows the app to organize files into a clean hierarchy even when the original structure was super messy without doubling the required disk space

Coordinate Standard (RAS+): The app forces all NIfTI volumes into RAS+ (Right, Anterior, Superior) orientation.

Why? nnUNet and most AI training pipelines require consistent voxel orientations to learn spatial relationships correctly. Also RAS is the standard for NIFTIs. 

Logic: If the input is detected as LPS (Left, Posterior, Superior), it is reoriented using nibabel.orientations before being saved to the NIFTI/ folder.

## 2. ROI Processing & "Cutting"
For large volumes (like whole-body CTs), the app provides a cut_volume utility to reduce VRAM usage and speed up inference as the sliding window overlap goes cubic. So getting a good cut reduces compute time by a lot. Leave enough space around so the model has enough context.  

Coordinate System: The GUI allows users to input bounds in either RAS (native to the app and e.g. 3d Slicer) or LPS (native to DICOM viewers like MicroDicom).

Destructive vs. Non-Destructive: If "Keep Originals" is checked, the cropped versions are saved to NIFTI_cut/, leaving the standard NIFTI/ folder intact for other tasks.

X-Axis Masking: For bilateral tasks (e.g., separating a left and right femur), the app uses a masking approach. Instead of physically splitting the file, it zeros out the voxels of the "undesired" side, which preserves the global coordinate alignment. This makes HU analysis easier and is not destrcutive to the OG NIFTI files. 

## 3. The nnUNet Inference Engine
The app utilizes the nnUNetv2 framework, the gold standard for medical image segmentation when VRAM is limited. 

The Predictor: The nnUNetPredictor is initialized with specific "folds" (usually 0 through 4). The final mask is an ensemble average of these 5 models, which significantly reduces "false positive" segmentations.

Cascade Architecture: For complex anatomy like the spine (Dataset 217), a multi-stage approach is used:

Stage 1: Predicts a low-resolution mask (label_lowres/) to find the general location and has better global context. 

Stage 2: Uses the low-res mask as a "anchor" to predict the high-resolution final segmentation.

Model Weights: Weights are stored in the directory defined by the nnUNet_results environment variable. 

Seperate file just for nnUNet and how to integrate new models is also provided. 

## 4. 3D Surface Reconstruction (multi_stl.py)
Converting a voxel mask into a 3D mesh requires balancing anatomical accuracy with surface smoothness.
### Conversion 
Marching Cubes: We use the skimage.measure.marching_cubes algorithm to generate the initial triangular mesh.
There are other algorithms but marching cubes is the standard and is not super compute intense. 
At the end we need the affine matrix of each Scan to get back to world corrdinates from the voxel space. This is done with 
```
verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
verts = (affine @ verts.T).T[:, :3]
verts = convert_to_LPS(verts)
``` 
### Smoothing 
Taubin Smoothing: Unlike standard Laplacian smoothing which "shrinks" volume, Taubin smoothing uses a low-pass filter to remove "stair-step" artifacts without altering the underlying volume.
The factor behaves inverted, so lower factor is stronger smoothing. Consult official documentation if needed. 

### Mesh Repair:

PyMeshFix: Automatically fills holes and ensures the mesh is "watertight" (manifold) for 3D printing.
It aslo caps open meshes which happen if we the segment goes to the boundary of the image. 

Island Removal: Optionally removes small floating artifacts (stray voxels) to keep only the largest continuous structure. Disable when using it with multi part structures like ribs. 


## 5. Multiprocessing and -threading 
To handle the significant computational load of medical imaging without freezing the user interface, this application employs a hybrid concurrency model. It distinguishes between I/O-bound tasks (waiting for the hard drive) and CPU-bound tasks (heavy mathematical calculations).

### Threading for UI Fluidity and I/O
The GUI itself runs on a single main thread. To prevent the "Not Responding" window hang, the app uses threading.Thread for background operations that involve waiting:

#### GUI Responsiveness: 
The main processing loop (process_data) is launched in a daemon thread so the user can still interact with the window while the AI works.

Indicator Scanning: When you select an input folder, a background thread scans DICOM headers to populate the "Scan Indicators" menu without stuttering the UI.

#### DICOM Linking: 
During the DICOM_splitter phase, a ThreadPoolExecutor is used to create hard links. Since linking is an I/O operation, multiple threads can queue these requests to the operating system simultaneously.

### Multiprocessing for Heavy Computation
Python’s Global Interpreter Lock (GIL) prevents multiple threads from performing CPU-heavy math at the same time. To bypass this, the app uses multiprocessing (specifically ProcessPoolExecutor) to utilize nearly every core on your workstation:

- **Parallel DICOM to NIfTI**: Conversion of multiple series happens in parallel, with each process handling a different scan.

- **Batched STL Generation**: Surface reconstruction is the most CPU-intensive part of the pipeline. The ParallelSTLProcessor breaks the list of labelmaps into Batches.
 Each batch is distributed across max_workers (defaulting to CPU count minus 4 to keep the system usable).
Each process independently runs the Marching Cubes, Taubin Smoothing, and PyMeshFix repair algorithms.
- **HU Analytics** : Calculating statistics (Mean, Skewness, Kurtosis) for millions of voxels is parallelized across subjects to speed up the generation of the final Excel report.

### Inter-Process Communication (IPC)
Because background processes cannot directly "talk" to the GUI, the app uses a queue.Queue:

Workers put ProgressEvent objects into the queue.

The GUI "polls" this queue every 100ms using root.after() to update the progress bar and status text safely on the main thread.


## 6. Data Integrity & Validation (Pydantic)
While you may not need to edit the core code, the application uses Pydantic (a data-validation library) to ensure that the settings you choose in the GUI are "sane" before the script starts.

**Fail-Fast Logic**: If you accidentally enter a negative number for a crop bound or an invalid email format, Pydantic catches this immediately and prevents the script from starting. This saves hours of processing time that would otherwise be wasted on a "broken" run.

**Traceability**: This is the engine behind the run_parameter.json file. It converts the complex state of the GUI into a clean, structured, and reproducible data format.

**Type Safety**: It ensures that variables remain the correct type (e.g., ensuring a coordinate is always an Integer and not a String). 


## 🔗 External Documentation & Libraries
To modify the core geometry or inference logic, refer to these libraries:

nnUNetv2 Official Docs: Handling training, inference, and environment variables.
https://github.com/MIC-DKFZ/nnUNet

PyMeshFix Docs: Deep dive into the repair() algorithms. (Cite as well when using.)
https://github.com/pyvista/pymeshfix

PyVista: Used for the Taubin and Laplacian smoothing implementations.
https://docs.pyvista.org/index.html

Nibabel: Documentation on handling NIfTI headers and affine matrices.
https://nipy.org/nibabel/

Pydantic: Type safety 
https://pydantic.dev/docs/validation/latest/get-started/

MLflow: Tracking server for experiments, not used to serve models.
Run the server with tmux in WSL, maybe use cloudfares quicktunnels for mobile access (use the auth then otherwise run without auth saves you a lot of headaches). 
This allows nicer insights as we can get graphs for each class and not just the avg also nice for system metrics, really finegrained CPU and GPU util. 
Also quick tunnels seem to have a rate limit, so after a restart it will not allow to get a new quick tunnel set up, it throws a generic error. Took me 1h too find out that they have not fixed that (issue since 2023 or so). 
https://mlflow.org/

Streamlit: for the dashboard, made UI super easy, needs to be run as its own process
```
python_exe = sys.executable
cmd = [python_exe, "-m", "streamlit", "run", "utils/streamlit_dbscan.py", "--", cleaned_metapath]
subprocess.run(cmd)
```

https://streamlit.io/


## File Traceability
Every run generates a decoder.json and a run_parameter.json.

decoder.json: Critical for clinical research. It maps the nnUNet filenames (e.g., CASE_001.nii.gz) back to the original unique names generated in sorting.

run_parameter.json: Stores the exact GUI configuration used, allowing any result to be perfectly replicated later.