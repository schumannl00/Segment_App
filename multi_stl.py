import nibabel as nib
import numpy as np
from skimage import measure
from scipy import ndimage
import pyvista as pv
from stl import mesh
import os 
from scipy.ndimage import binary_fill_holes, binary_closing
import pymeshfix
import logging
import json 
import shutil
import trimesh 
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from utils.stl_metadata import calculate_volume_and_surface_area, save_metadata_to_json
from typing import List, Dict, Tuple, Optional, Union, Any, TypedDict, Literal 
from pydantic import BaseModel, Field, DirectoryPath, field_validator
from abc import ABC, abstractmethod
import time


# Define a shortcut for types objects, depre
class SegmentConfig(TypedDict, total=False):
    label: str
    smoothing: float
    mesh_smoothing_method: str
    mesh_smoothing_iterations: int
    mesh_smoothing_factor: float

PathLike = Union[str, Path]
ProcessResult = Tuple[str, bool, Optional[str], List[Tuple[str, Dict[str, float]]]]


class MeshSmoothingConfig(BaseModel):
    method: Literal['taubin', 'laplacian'] = 'taubin'
    iterations: int = Field(default=100, ge=0)
    factor: float = Field(default=0.1, gt=0)
    
    
class LabelConfig(BaseModel):
    label_name: str
    volume_smoothing: float = Field(default=1.0, ge=0)
    mesh_config: MeshSmoothingConfig = Field(default_factory=MeshSmoothingConfig)
    
class STLProcessingConfig(BaseModel):
    input_dir: DirectoryPath
    output_root: Path
    segment_params: Dict[int, LabelConfig]
    fill_holes: int = 0
    use_pymeshfix: bool = True
    remove_islands: bool = True
    max_workers: Optional[int] = 12
    batch_size: int = Field(default=50, gt=0)
    resume: bool = True
    stl_metadata_path: Optional[Path] = None
    split: bool = False
    
    
class STLTask(BaseModel):
    """Container for a single NIfTI file task to be pickled across processes."""
    input_file: Path
    output_dir: Path
    file_number: str
    
    class Config:
        arbitrary_types_allowed = True
    
    
class BaseSurfaceReconstructor(ABC):
    def __init__(self, config: STLProcessingConfig):
        self.config = config

    @abstractmethod
    def run(self):
        """Orchestrate the directory processing and batching."""
        pass

    @staticmethod
    @abstractmethod
    def process_file(task: STLTask, config: STLProcessingConfig) -> tuple:
        """The core Marching Cubes and smoothing logic."""
        pass
    
   
    

def debug_normals(verts, faces, mag=5.0):
    """Visualizes face normals as arrows."""
    # Create the PyVista mesh
    # Note: faces in PyVista need to be [3, v1, v2, v3, 3, v4, v5, v6...]
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    mesh = pv.PolyData(verts, pv_faces)
    
    # Compute and plot normals
    # 'mag' controls arrow length; 'use_every' prevents clutter
    mesh.plot_normals(mag=mag, faces=True, show_edges=True, use_every=10, color='red')

#Here the laplacian smooothing should b e avoided, unless in testing weird edge casing to perform an opening, if binary operning does not work, test laplacian, the shrinkage might do it
def smooth_mesh_pyvista(vertices : np.ndarray, faces : np.ndarray, method : str ='taubin', n_iter : int =100, relaxation_factor : float =0.1) -> np.ndarray:
    """Smooth a mesh using PyVista's smoothing algorithms."""
    vertices = np.array(vertices, dtype=np.float64)
    original_centroid = np.mean(vertices, axis=0)
    print(f"Original centroid located at {original_centroid}")
    
    faces_pv = np.hstack([[3, f[0], f[1], f[2]] for f in faces])
    mesh_pv : pv.PolyData  = pv.PolyData(vertices, faces_pv)
    mesh_smoothed: pv.PolyData 
    temp_mesh: Any = mesh_pv
    if method == 'laplacian':
        mesh_smoothed = temp_mesh.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor)
    elif method == 'taubin':
        mesh_smoothed = temp_mesh.smooth_taubin(n_iter=n_iter, pass_band=relaxation_factor)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    smoothed_vertices = mesh_smoothed.points.astype(np.float64)
    new_centroid = np.mean(smoothed_vertices, axis=0)
    shift = original_centroid - new_centroid
    print(f"Correcting centroid shift of {shift}.")
    smoothed_vertices += shift
    return smoothed_vertices



def fill_holes_3d(segmentation):
    """Fill small holes in a 3D binary segmentation mask."""
    filled_segmentation = np.zeros_like(segmentation, dtype=bool)
    
    for i in range(segmentation.shape[0]):
        filled_segmentation[i] = binary_fill_holes(segmentation[i])
    
    for i in range(segmentation.shape[1]):
        filled_segmentation[:, i] = binary_fill_holes(filled_segmentation[:, i])
    
    for i in range(segmentation.shape[2]):
        filled_segmentation[:, :, i] = binary_fill_holes(filled_segmentation[:, :, i])
    
    structure = np.ones((3, 3, 3))
    filled_segmentation = binary_closing(filled_segmentation, structure=structure)
    print('Filling mesh')
    
    return filled_segmentation.astype(np.uint8)

# do not touch this, unless messing with the underlying coordinate system, if stuff looks odd in the stl, this is the palce to fix that 
def convert_to_LPS(vertices : np.ndarray) -> np.ndarray:
    """Convert vertices to LPS coordinate system."""
    vertices[:, 1] *= -1.0
    vertices[:, 0] *= -1.0
    return vertices


def process_single_file(file_info : Tuple[str, str, str | int ], segment_params : Dict[int, SegmentConfig], fill_holes : int = 0, use_pymeshfix : bool = True, remove_islands : bool = True) -> ProcessResult:
    """
    Process a single NIfTI file. This function is designed to be called by multiprocessing.
    
    Parameters:
    file_info (tuple): (input_file, output_dir, file_number)
    segment_params (dict): Segment processing parameters
    fill_holes (int): Whether to fill holes in the segmentation
    use_pymeshfix (bool): Whether to use pymeshfix for repair
    remove_islands (bool): argument for pymeshfix.repair
    Returns:
    tuple: (file_name, success, error_message)
    """
    input_file, output_dir, file_number = file_info
    file_name = os.path.basename(input_file)
    metadata_entries = []
    try:
        simple_name = Path(file_name).stem.split('.')[0]
        print(f"[PID {os.getpid()}] Processing {file_name}")
        nib.openers.Opener.default_compresslevel = 9 #type: ignore
        nii_img = nib.load(input_file) #type: ignore
        nii_data = nii_img.get_fdata() #type: ignore
        spacing = nii_img.header.get_zooms() #type: ignore
        affine = nii_img.affine #type: ignore
        #orientation = nib.orientations.io_orientation(affine)
        #print(" Image orientation:", orientation)
        for label, params in segment_params.items():
            volume_smoothing = params.get('smoothing', 0.1)
            output_label = params.get('label', f"segment_{label}")
            mesh_method = params.get('mesh_smoothing_method', 'taubin')
            mesh_iterations = params.get('mesh_smoothing_iterations', 100)  #100 seems to be the default value from literature, works in most cases, and going down to 50 does not make a diffrence
            mesh_factor = params.get('mesh_smoothing_factor', 0.1)  #lower -->  stronger smoothing, it is a low pass band filter 
            
            # Extract binary segment
            binary_segment = (nii_data == int(label))
            segment_voxel_count = np.sum(binary_segment)
            #This is just for debugging, normally we should get nice ints, but these are our workarounds if something goes wrong in memory 
            if segment_voxel_count == 0:
                binary_segment = (np.round(nii_data).astype(int) == int(label))
                segment_voxel_count = np.sum(binary_segment)
            
            if segment_voxel_count == 0:
                binary_segment = np.isclose(nii_data, float(label), rtol=1e-5, atol=1e-8)
                segment_voxel_count = np.sum(binary_segment)
            
            print(f"[PID {os.getpid()}] Segment {label} voxel count: {segment_voxel_count}")
            
            if segment_voxel_count == 0:
                print(f"[PID {os.getpid()}] No voxels found for label {label}")
                continue
            
            if fill_holes > 0:
                binary_segment = fill_holes_3d(binary_segment)
            
            # Smooth and threshold
            smoothed_segment = ndimage.gaussian_filter(binary_segment.astype(float), sigma=volume_smoothing)  #removes small odd pointy mislabeling, these cause spkies in the stl, which cause probolem in downstream tasks, as it causes exploding gradients 
            binary_smooth = smoothed_segment > 0.5
            
            if np.sum(binary_smooth) == 0:
                print(f"[PID {os.getpid()}] No voxels after smoothing for label {label}")  #good thing if it was just a noise spot in multiclass seg
                continue
            
            # Marching cubes
            try:
                verts, faces, _, _ = measure.marching_cubes(binary_smooth, level=0.5)  #seems to align with what 3d slicer does 
                print(f"[PID {os.getpid()}] Marching cubes: {len(verts)} vertices, {len(faces)} faces")
            except Exception as e:
                print(f"[PID {os.getpid()}] Marching cubes failed for label {label}: {e}")
                continue
            
            # Apply affine transformation. This converts voxel coordinates to world coordinates 
            verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
            verts = (affine @ verts.T).T[:, :3]
            convert_to_LPS(verts)
            
            # 2. Initial Heavy Smoothing
            # Providing a cleaner surface for pymeshfix
            if mesh_iterations > 0:
                verts = smooth_mesh_pyvista(verts, faces, method=mesh_method, 
                                          n_iter=mesh_iterations, relaxation_factor=mesh_factor)
            
            # 3. Pymeshfix Repair and Normal Correction
            if use_pymeshfix:
                try:
                    meshfix = pymeshfix.MeshFix(verts, faces)
                    meshfix.repair(verbose=False, remove_smallest_components=remove_islands)
                    verts, faces = meshfix.points, meshfix.faces
                    
                    t_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    
                    # Fix consistency (ensure all triangles face the same way they should already, if not watertight might make more problems)
                    #trimesh.repair.fix_normals(t_mesh)
                    
                    trimesh.repair.fix_inversion(t_mesh)
                    
                    verts, faces = t_mesh.vertices, t_mesh.faces
                   

                    # 4. Light post-repair smoothing (n=50) to remove pymeshfix artifacts
                    verts = smooth_mesh_pyvista(verts, faces, method=mesh_method, 
                                              n_iter=50, relaxation_factor=0.1)
                except Exception as e:
                    print(f"[PID {os.getpid()}] Orientation correction failed: {e}")


            # uncomment if somethin looks wrong with the normals again, but only for testing, will otherwise stop/ stall the loop 
            #debug_normals(verts, faces)   

            
            # Calculate volume and surface area
            volume_mm3, surface_area_mm2 = calculate_volume_and_surface_area(verts, faces)
            #print(volume_mm3, surface_area_mm2)
            simple_name_ = f"{simple_name}_{output_label}"
            metadata_entries.append((simple_name_, {"Mesh_volume_mm3" : volume_mm3, "Surface_Area_mm2": surface_area_mm2}))
            stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, face in enumerate(faces):
                for j in range(3):
                    stl_mesh.vectors[i][j] = verts[face[j]] #type: ignore
            
            if "links" in output_dir:
                output_file = f"{output_dir}/{output_label}_{file_number}_links.stl"
            elif 'rechts' in output_dir:
                output_file = f"{output_dir}/{output_label}_{file_number}_rechts.stl"
            else:
                output_file = f"{output_dir}/{output_label}_{file_number}.stl"
            
            stl_mesh.save(output_file) #type: ignore 
            print(f"[PID {os.getpid()}] Saved {output_file}")
        
        return (file_name, True, None, metadata_entries)
    
    except Exception as e:
        error_msg = f"Failed to process {file_name}: {str(e)}"
        print(f"[PID {os.getpid()}] {error_msg}")
        return (file_name, False, error_msg, metadata_entries)


def load_checkpoint(checkpoint_file):
    """Load checkpoint of completed files."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_checkpoint(checkpoint_file, completed, failed):
    """Save checkpoint of completed and failed files."""
    with open(checkpoint_file, 'w') as f:
        json.dump({"completed": completed, "failed": failed}, f, indent=2)


def process_directory_parallel(input_dir : PathLike, output_root_dir : PathLike , segment_params : dict, 
                               fill_holes : int =0, split: bool=False, use_pymeshfix : bool =True, remove_islands : bool =True, 
                               max_workers : int | None  =None, batch_size : int =50, resume : bool =True, stl_metadata_path : PathLike | None = None) -> List[ProcessResult]:
    
    os.makedirs(output_root_dir, exist_ok=True)
    p = Path(output_root_dir)
    checkpoint_file = os.path.join(p.parent, ".stl_processing_checkpoint.json")

    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file) if resume else {"completed": [], "failed": []}
    completed_files = set(checkpoint["completed"])
    all_failed = checkpoint["failed"].copy()
    
    # Prepare list of files to process
    all_file_tasks = []
    
    for file_name in sorted(os.listdir(input_dir)):  # Sort for consistent ordering
        if file_name.endswith(".nii.gz"):
            # Skip if already completed
            if file_name in completed_files:
                print(f"⊙ Skipping (already completed): {file_name}")
                continue
            
            input_file = os.path.join(input_dir, file_name)
            file_number = file_name.split('_')[1].split('.')[0].split("-")[0]
            
            output_dir = os.path.join(output_root_dir, f"STL{file_number}")
            
            if split:
                side = file_name.split('_')[1].split('.')[0].split("-")[1]
                if side == "links":
                    output_dir += "_links"
                elif side == "rechts":
                    output_dir += "_rechts"
            
            os.makedirs(output_dir, exist_ok=True)
            all_file_tasks.append((input_file, output_dir, file_number))
    
    remaining_files = len(all_file_tasks)
    total_files = remaining_files + len(completed_files)
    print(f"\n{'='*60}")
    print(f"Total files: {total_files}")
    print(f"Already completed: {len(completed_files)}")
    print(f"Remaining to process: {remaining_files}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {max_workers or mp.cpu_count()}")
    print(f"{'='*60}\n")
    
    if total_files == 0:
        print("No files to process!")
        return []
    
    # Create partial function with fixed parameters
    process_func = partial(process_single_file, 
                          segment_params=segment_params,
                          fill_holes=fill_holes,
                          use_pymeshfix=use_pymeshfix,
                          remove_islands=remove_islands)
    
    # Process in batches
    all_results = []
    num_batches = (total_files + batch_size - 1) // batch_size
    stl_metadata = {}

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_tasks = all_file_tasks[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx + 1}/{num_batches}")
        print(f"Files {start_idx + 1}-{end_idx} of {total_files}")
        print(f"{'='*60}\n")
        
        batch_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks in this batch
            futures = {executor.submit(process_func, task): task for task in batch_tasks}
            
            # Collect results as they complete
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                    file_name, success, error, metadata_entires = result
                    if success:
                        print(f"✓ Completed: {file_name}")
                        completed_files.add(file_name)
                        for name, meta in metadata_entires:
                            stl_metadata[name] = meta
                    else:
                        print(f"✗ Failed: {file_name} - {error}")
                        all_failed.append({"file": file_name, "error": error})
                except Exception as e:
                    print(f"✗ Exception for {os.path.basename(task[0])}: {str(e)}")
                    batch_results.append((os.path.basename(task[0]), False, str(e), []))
                    all_failed.append({"file": os.path.basename(task[0]), "error": str(e)})
        
        all_results.extend(batch_results)
        
        # Save checkpoint after each batch, that is the only real point for the batches, delete the checkpoint if rerunning on old data 
        save_checkpoint(checkpoint_file, list(completed_files), all_failed)
        
        # Batch summary
        batch_success = sum(1 for entry in batch_results if entry[1])
        print(f"\nBatch {batch_idx + 1} complete: {batch_success}/{len(batch_results)} succeeded")
        print(f"Overall progress: {len(completed_files)}/{total_files} total files")
    
    # Final summary
    successful = sum(1 for entry in  all_results if entry[1])
    failed = len(all_results) - successful
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(completed_files)}")
    print(f"{'='*60}")
    
    if all_failed:
        print(f"\nFailed files saved to: {checkpoint_file}")
    
    if stl_metadata_path:
        save_metadata_to_json(stl_metadata, stl_metadata_path)
        print(f"STL metadata saved to: {stl_metadata_path}")

    return all_results


class ParallelSTLProcessor(BaseSurfaceReconstructor):
    def run(self):
        start_time = time.time()
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.output_root.parent / ".stl_processing_checkpoint.json"
        
        # 1. Checkpoint & Task Preparation
        checkpoint = self._load_checkpoint(checkpoint_path)
        all_tasks = self._prepare_tasks(checkpoint["completed"])
        
        if not all_tasks:
            print("No files left to process.")
            return

        print(f"Starting STL conversion for {len(all_tasks)} files...")
        stl_metadata = {}
        all_failed = checkpoint["failed"]

        # 2. Batch Execution
        for i in range(0, len(all_tasks), self.config.batch_size):
            batch = all_tasks[i : i + self.config.batch_size]
            print(f"\nProcessing Batch {(i // self.config.batch_size) + 1}...")
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.process_file, task, self.config): task for task in batch}
                
                for future in as_completed(futures):
                    file_name, success, error, metadata_entries = future.result()
                    if success:
                        checkpoint["completed"].append(file_name)
                        for name, meta in metadata_entries:
                            stl_metadata[name] = meta
                        print(f"✓ Completed: {file_name}")
                    else:
                        all_failed.append({"file": file_name, "error": error})
                        print(f"✗ Failed: {file_name}")

            # 3. Save progress after each batch
            self._save_checkpoint(checkpoint_path, checkpoint["completed"], all_failed)

        # 4. Final Metadata Export
        if self.config.stl_metadata_path and stl_metadata:
            from utils.stl_metadata import save_metadata_to_json
            save_metadata_to_json(stl_metadata, self.config.stl_metadata_path)

        print(f"\n{'='*30}\nTotal execution time: {time.time() - start_time:.2f}s")

    @staticmethod
    def process_file(task: STLTask, config: STLProcessingConfig) -> tuple:
        """Process a single NIfTI file to STL, returning success status and metadata."""
        file_name = task.input_file.name
        metadata_entries = []
        try:
            simple_name = task.input_file.stem.split('.')[0]
            nii_img = nib.load(str(task.input_file))
            nii_data = nii_img.get_fdata()
            affine = nii_img.affine
            
            for label, params in config.segment_params.items():
                # Extract Binary Segment
                binary_segment = (nii_data == int(label))
                if np.sum(binary_segment) == 0:
                    binary_segment = (np.round(nii_data).astype(int) == int(label))
                
                if np.sum(binary_segment) == 0:
                    continue

                if config.fill_holes > 0:
                    binary_segment = fill_holes_3d(binary_segment) # Helper function, works ok should be tunred off by default 

                # Volume Smoothing
                smoothed = ndimage.gaussian_filter(binary_segment.astype(float), sigma=params.volume_smoothing)
                binary_smooth = smoothed > 0.5
                
                if np.sum(binary_smooth) == 0: continue

                # Marching Cubes
                verts, faces, _, _ = measure.marching_cubes(binary_smooth, level=0.5)
            
                # Voxel to World + LPS Conversion
                verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
                verts = (affine @ verts.T).T[:, :3]
                verts = convert_to_LPS(verts) # Helper function to get coordinate system right 

                # Mesh Smoothing
                m_cfg = params.mesh_config
                if m_cfg.iterations > 0:
                    verts = smooth_mesh_pyvista(verts, faces, method=m_cfg.method, 
                                               n_iter=m_cfg.iterations, relaxation_factor=m_cfg.factor)

                # Pymeshfix Repair
                if config.use_pymeshfix:
                    meshfix = pymeshfix.MeshFix(verts, faces)
                    meshfix.repair(remove_smallest_components=config.remove_islands)
                    verts, faces = meshfix.points, meshfix.faces
                    
                    t_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    trimesh.repair.fix_inversion(t_mesh)
                    verts, faces = t_mesh.vertices, t_mesh.faces
                    
                    # Post-repair polish
                    verts = smooth_mesh_pyvista(verts, faces, method=m_cfg.method, n_iter=50, relaxation_factor=0.1)
                    
                    
                #debug_normals(verts, faces)   

                # Export STL
                from utils.stl_metadata import calculate_volume_and_surface_area
                vol, surf = calculate_volume_and_surface_area(verts, faces)
                metadata_entries.append((f"{simple_name}_{params.label_name}", {"Mesh_volume_mm3": vol, "Surface_Area_mm2": surf}))

                stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, face in enumerate(faces):
                    for j in range(3):
                        stl_mesh.vectors[i][j] = verts[face[j]]
                
                out_name = f"{params.label_name}_{task.file_number}.stl"
                stl_mesh.save(task.output_dir / out_name)
            
            return (file_name, True, None, metadata_entries)
        except Exception as e:
            return (file_name, False, str(e), [])

    def _prepare_tasks(self, completed_list) -> List[STLTask]:
        tasks = []
        completed_set = set(completed_list)
        for f_name in sorted(os.listdir(self.config.input_dir)):
            if f_name.endswith(".nii.gz") and f_name not in completed_set:
                input_path = self.config.input_dir / f_name
                file_num = f_name.split('_')[1].split('.')[0].split("-")[0]
                
                out_dir = self.config.output_root / f"STL{file_num}"
                if self.config.split and "-" in f_name:
                    side = f_name.split('_')[1].split('.')[0].split("-")[1]
                    out_dir = Path(str(out_dir) + f"_{side}")
                
                out_dir.mkdir(parents=True, exist_ok=True)
                tasks.append(STLTask(input_file=input_path, output_dir=out_dir, file_number=file_num))
        return tasks

    def _load_checkpoint(self, path):
        if self.config.resume and path.exists():
            return json.loads(path.read_text())
        return {"completed": [], "failed": []}

    def _save_checkpoint(self, path, completed, failed):
        path.write_text(json.dumps({"completed": completed, "failed": failed}, indent=2))

# Example segment parameters
segment_params_leg = {
    1: {
        'label': "Fibula",
        'smoothing': 0.2,  
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 250,  
        'mesh_smoothing_factor': 0.1  
    },
    2: {
        'label': "Patella",
        'smoothing': 0.2,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 250, 
        'mesh_smoothing_factor': 0.1  
    },
    3: {
        'label': "Tibia",
        'smoothing': 0.2,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 250,  
        'mesh_smoothing_factor': 0.1
    },
    4: {
        'label': "Femur",
        'smoothing': 0.2,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 250,  
        'mesh_smoothing_factor': 0.1
    }
}

segment_ankle = {
    1: {
        'label': "Fibula",
        'smoothing': 1.0,  
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 150,  
        'mesh_smoothing_factor': 0.1  
    },
    2: {
        'label': "Talus" ,
        'smoothing': 1.0,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 150, 
        'mesh_smoothing_factor': 0.1  
    },
    3: {
        'label': "Tibia",
        'smoothing': 1.0,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 150,  
        'mesh_smoothing_factor': 0.1
}
}


spinal_params = {
    # Cervical Vertebrae (C1 - C7)
    1: LabelConfig(label_name="C1_Atlas", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    2: LabelConfig(label_name="C2_Axis", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    3: LabelConfig(label_name="C3", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    4: LabelConfig(label_name="C4", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    5: LabelConfig(label_name="C5", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    6: LabelConfig(label_name="C6", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    7: LabelConfig(label_name="C7", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),

    # Thoracic Vertebrae (T1 - T12)
    8: LabelConfig(label_name="T1", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    9: LabelConfig(label_name="T2", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    10: LabelConfig(label_name="T3", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    11: LabelConfig(label_name="T4", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    12: LabelConfig(label_name="T5", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    13: LabelConfig(label_name="T6", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    14: LabelConfig(label_name="T7", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    15: LabelConfig(label_name="T8", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    16: LabelConfig(label_name="T9", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    17: LabelConfig(label_name="T10", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    18: LabelConfig(label_name="T11", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    19: LabelConfig(label_name="T12", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),

    # Lumbar Vertebrae (L1 - L6)
    20: LabelConfig(label_name="L1", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    21: LabelConfig(label_name="L2", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    22: LabelConfig(label_name="L3", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    23: LabelConfig(label_name="L4", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    24: LabelConfig(label_name="L5", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
    25: LabelConfig(label_name="L6", volume_smoothing=0.5, mesh_config=MeshSmoothingConfig(iterations=150)),
}


if __name__ == "__main__":
   
    mp.freeze_support()
    
    config = STLProcessingConfig(
        input_dir=Path(r"C:\Users\schum\Desktop\zesbo\scoliosis_project\labels"),
        output_root=Path(r"C:\Users\schum\Desktop\zesbo\scoliosis_project\stl_output"),
        segment_params=spinal_params,
        max_workers=mp.cpu_count() - 4,
        batch_size=10,
        stl_metadata_path=Path(r"C:\Users\schum\Desktop\zesbo\scoliosis_project\stl_metadata.json")
        )
    
    
    processor = ParallelSTLProcessor(config)
    processor.run()
        