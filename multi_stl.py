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


def smooth_mesh_pyvista(vertices, faces, method='taubin', n_iter=100, relaxation_factor=0.1):
    """Smooth a mesh using PyVista's smoothing algorithms."""
    vertices = np.array(vertices, dtype=np.float64)
    original_centroid = np.mean(vertices, axis=0)
    print(f"Original centroid located at {original_centroid}")
    
    faces_pv = np.hstack([[3, f[0], f[1], f[2]] for f in faces])
    mesh_pv = pv.PolyData(vertices, faces_pv)
    
    if method == 'laplacian':
        mesh_smoothed = mesh_pv.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor)
    elif method == 'taubin':
        mesh_smoothed = mesh_pv.smooth_taubin(n_iter=n_iter, pass_band=relaxation_factor)
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


def convert_to_LPS(vertices):
    """Convert vertices to LPS coordinate system."""
    vertices[:, 1] = -vertices[:, 1]
    vertices[:, 0] *= -1.0
    return vertices


def process_single_file(file_info, segment_params, fill_holes=0, use_pymeshfix=True, remove_islands=True):
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
    
    try:
        print(f"[PID {os.getpid()}] Processing {file_name}")
        
        nib.openers.Opener.default_compresslevel = 9
        nii_img = nib.load(input_file)
        nii_data = nii_img.get_fdata()
        spacing = nii_img.header.get_zooms()
        affine = nii_img.affine
        
        for label, params in segment_params.items():
            volume_smoothing = params.get('smoothing', 0.1)
            output_label = params.get('label', f"segment_{label}")
            mesh_method = params.get('mesh_smoothing_method', 'taubin')
            mesh_iterations = params.get('mesh_smoothing_iterations', 100)
            mesh_factor = params.get('mesh_smoothing_factor', 0.1)
            
            # Extract binary segment
            binary_segment = (nii_data == int(label))
            segment_voxel_count = np.sum(binary_segment)
            
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
            smoothed_segment = ndimage.gaussian_filter(binary_segment.astype(float), sigma=volume_smoothing)
            binary_smooth = smoothed_segment > 0.5
            
            if np.sum(binary_smooth) == 0:
                print(f"[PID {os.getpid()}] No voxels after smoothing for label {label}")
                continue
            
            # Marching cubes
            try:
                verts, faces, _, _ = measure.marching_cubes(binary_smooth, level=0.5)
                print(f"[PID {os.getpid()}] Marching cubes: {len(verts)} vertices, {len(faces)} faces")
            except Exception as e:
                print(f"[PID {os.getpid()}] Marching cubes failed for label {label}: {e}")
                continue
            
            # Apply affine transformation
            verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
            verts = (affine @ verts.T).T[:, :3]
            convert_to_LPS(verts)
            
            # Mesh smoothing
            if mesh_iterations > 0:
                verts = smooth_mesh_pyvista(verts, faces, method=mesh_method, 
                                          n_iter=mesh_iterations, relaxation_factor=mesh_factor)
            
            # Pymeshfix repair
            if use_pymeshfix:
                print(f"[PID {os.getpid()}] Attempting Pymeshfix repair")
                try:
                    meshfix = pymeshfix.MeshFix(verts, faces)
                    meshfix.repair(verbose=False, remove_smallest_components=remove_islands)
                    verts_repaired = meshfix.v
                    faces_repaired = meshfix.f
                    if faces_repaired.shape[0] > 0:
                        print(f"[PID {os.getpid()}] Pymeshfix successful")
                        verts, faces = verts_repaired, faces_repaired
                    else:
                        print(f"[PID {os.getpid()}] Pymeshfix resulted in empty mesh")
                except Exception as e:
                    print(f"[PID {os.getpid()}] Pymeshfix failed: {e}")
                
                verts = smooth_mesh_pyvista(verts, faces, method=mesh_method, 
                                          n_iter=100, relaxation_factor=0.1)
            
            # Save STL
            stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, face in enumerate(faces):
                for j in range(3):
                    stl_mesh.vectors[i][j] = verts[face[j]]
            
            if "links" in output_dir:
                output_file = f"{output_dir}/{output_label}_{file_number}_links.stl"
            elif 'rechts' in output_dir:
                output_file = f"{output_dir}/{output_label}_{file_number}_rechts.stl"
            else:
                output_file = f"{output_dir}/{output_label}_{file_number}.stl"
            
            stl_mesh.save(output_file)
            print(f"[PID {os.getpid()}] Saved {output_file}")
        
        return (file_name, True, None)
    
    except Exception as e:
        error_msg = f"Failed to process {file_name}: {str(e)}"
        print(f"[PID {os.getpid()}] {error_msg}")
        return (file_name, False, error_msg)


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


def process_directory_parallel(input_dir, output_root_dir, segment_params, 
                               fill_holes=0, split=False, use_pymeshfix=True, remove_islands=True, 
                               max_workers=None, batch_size=50, resume=True):
    """
    Process directory with parallel execution at the file level, using batching.
    
    Parameters:
    input_dir (str): Input directory containing .nii.gz files
    output_root_dir (str): Root output directory for STL files
    segment_params (dict): Segment processing parameters
    fill_holes (int): Whether to fill holes
    split (bool): Whether to split by side (links/rechts)
    use_pymeshfix (bool): Whether to use pymeshfix
    remove_islands (bool): argument for remove_smallest_components of pymeshfix.repair 
    max_workers (int): Maximum number of parallel workers (default: CPU count)
    batch_size (int): Number of files to process per batch (default: 50)
    resume (bool): Whether to resume from checkpoint (default: True)
    """
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
                    file_name, success, error = result
                    if success:
                        print(f"✓ Completed: {file_name}")
                        completed_files.add(file_name)
                    else:
                        print(f"✗ Failed: {file_name} - {error}")
                        all_failed.append({"file": file_name, "error": error})
                except Exception as e:
                    print(f"✗ Exception for {os.path.basename(task[0])}: {str(e)}")
                    batch_results.append((os.path.basename(task[0]), False, str(e)))
                    all_failed.append({"file": os.path.basename(task[0]), "error": str(e)})
        
        all_results.extend(batch_results)
        
        # Save checkpoint after each batch
        save_checkpoint(checkpoint_file, list(completed_files), all_failed)
        
        # Batch summary
        batch_success = sum(1 for _, success, _ in batch_results if success)
        print(f"\nBatch {batch_idx + 1} complete: {batch_success}/{len(batch_results)} succeeded")
        print(f"Overall progress: {len(completed_files)}/{total_files} total files")
    
    # Final summary
    successful = sum(1 for _, success, _ in all_results if success)
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
    
    return all_results


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
        'smoothing': 3.0,  
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 250,  
        'mesh_smoothing_factor': 0.1  
    },
    2: {
        'label': "Talus" ,
        'smoothing': 3.0,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 250, 
        'mesh_smoothing_factor': 0.1  
    },
    3: {
        'label': "Tibia",
        'smoothing': 3.0,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 250,  
        'mesh_smoothing_factor': 0.1
}
}


if __name__ == "__main__":
   
    mp.freeze_support()
    
    process_directory_parallel(
        input_dir=r"E:\stl_multi\label",
        output_root_dir=r"E:\stl_multi\stl",
        segment_params=segment_ankle,
        remove_islands=True,
        split=False,
        max_workers=14,      
        batch_size=10,       
        resume=True          
    )