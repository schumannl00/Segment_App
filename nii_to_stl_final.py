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


def smooth_mesh_pyvista(vertices, faces, method='taubin', n_iter=100, relaxation_factor=0.1):
    """
    Smooth a mesh using PyVista's smoothing algorithms.
    
    Parameters:
    vertices (np.array): Vertex coordinates
    faces (np.array): Face indices
    method (str): Smoothing method ('laplacian', 'taubin')
    n_iter (int): Number of smoothing iterations
    relaxation_factor (float): Relaxation factor for smoothing
    
    Returns:
    np.array: Smoothed vertices
    """

    vertices= np.array(vertices, dtype=np.float64)

    original_centroid= np.mean(vertices, axis=0)
    print(f"Original centroid located at {original_centroid}")
    # Create faces array in required format for PyVista
    faces_pv = np.hstack([[3, f[0], f[1], f[2]] for f in faces])
    
    # Create PyVista mesh
    mesh_pv = pv.PolyData(vertices, faces_pv)
    
    # Apply smoothing based on method
    if method == 'laplacian':
        mesh_smoothed = mesh_pv.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor)
    elif method == 'taubin':
        mesh_smoothed = mesh_pv.smooth_taubin(n_iter=n_iter, pass_band=relaxation_factor)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    smoothed_vertices=mesh_smoothed.points.astype(np.float64)
    new_centroid= np.mean(smoothed_vertices,axis=0)
    shift= original_centroid - new_centroid
    print(f"Correcting centroid shift of {shift}.")
    smoothed_vertices += shift
    return smoothed_vertices

def fill_holes_3d(segmentation):
    """
    Fill small holes in a 3D binary segmentation mask and apply binary closing.
    
    Parameters:
    segmentation (numpy array): 3D binary mask
    
    Returns:
    numpy array: 3D mask with holes filled and closed
    """
    filled_segmentation = np.zeros_like(segmentation, dtype=bool)
    
    # Apply binary_fill_holes slice-by-slice along each axis
    for i in range(segmentation.shape[0]):
        filled_segmentation[i] = binary_fill_holes(segmentation[i])
    
    for i in range(segmentation.shape[1]):
        filled_segmentation[:, i] = binary_fill_holes(filled_segmentation[:, i])
    
    for i in range(segmentation.shape[2]):
        filled_segmentation[:, :, i] = binary_fill_holes(filled_segmentation[:, :, i])
    
    # Apply binary closing to further remove small holes and noise
    structure = np.ones((3,3,3))  # Define structuring element size
    filled_segmentation = binary_closing(filled_segmentation, structure=structure)
    print('Filling mesh')
    
    return filled_segmentation.astype(np.uint8)  # Convert back to 0/1 format


def analyze_mesh_holes(vertices, faces, max_holes_to_print=10):
    """
    Analyze holes in a mesh and print information about their sizes.
    
    Parameters:
    vertices (np.array): Vertex coordinates
    faces (np.array): Face indices
    max_holes_to_print (int): Maximum number of holes to print information about
    
    Returns:
    dict: Statistics about holes in the mesh
    """
    # Create faces array in required format for PyVista
    faces_pv = np.hstack([[3, f[0], f[1], f[2]] for f in faces])
    
    # Create PyVista mesh
    mesh_pv = pv.PolyData(vertices, faces_pv)
    
    # Find holes
    holes = mesh_pv.extract_feature_edges(feature_edges=True, boundary_edges=True, 
                                         non_manifold_edges=False, manifold_edges=False)
    
    # Analyze holes
    if holes.n_points == 0:
        print("No holes found in the mesh.")
        return {"hole_count": 0}
    
    # Separate holes into different connected components
    hole_components = holes.split_bodies()
    
    # Calculate hole sizes
    hole_sizes = [component.length for component in hole_components]
    hole_sizes.sort(reverse=True)
    
    # Print hole statistics
    print(f"Found {len(hole_sizes)} holes in the mesh.")
    print(f"Largest hole size: {hole_sizes[0]:.2f}")
    print(f"Smallest hole size: {hole_sizes[-1]:.2f}")
    print(f"Average hole size: {sum(hole_sizes)/len(hole_sizes):.2f}")
    
    # Print details of largest holes
    print(f"\nLargest {min(max_holes_to_print, len(hole_sizes))} holes:")
    for i, size in enumerate(hole_sizes[:max_holes_to_print]):
        print(f"  Hole {i+1}: Size = {size:.2f}")
    
    return {
        "hole_count": len(hole_sizes),
        "largest_hole": hole_sizes[0],
        "smallest_hole": hole_sizes[-1],
        "average_hole_size": sum(hole_sizes)/len(hole_sizes),
        "hole_sizes": hole_sizes
    }

def fill_holes_pyvista(vertices, faces, hole_size=1000):
    """
    Fill holes in mesh using PyVista's implementation of VTK's fill holes algorithm.
    
    Parameters:
    vertices (np.array): Vertex coordinates
    faces (np.array): Face indices
    hole_size (float): Maximum size of holes to fill
    
    Returns:
    tuple: (vertices, faces) of the mesh with filled holes
    """
    # Create faces array in required format for PyVista
    faces_pv = np.hstack([[3, f[0], f[1], f[2]] for f in faces])
    
    # Create PyVista mesh
    mesh_pv = pv.PolyData(vertices, faces_pv)
    
    # Fill holes in the mesh
    filled_mesh = mesh_pv.fill_holes(hole_size)
    
    # Extract faces from filled mesh
    filled_faces = []
    faces_array = filled_mesh.faces
    i = 0
    while i < len(faces_array):
        n_points = faces_array[i]
        if n_points == 3:  # We only want triangular faces
            filled_faces.append([faces_array[i+1], faces_array[i+2], faces_array[i+3]])
        i += n_points + 1
    
    return filled_mesh.points, np.array(filled_faces)

def convert_nifti_to_stls(input_file, output_prefix, label_map=None, smoothing_factor=1.0, 
                         mesh_smoothing_method='taubin', mesh_smoothing_iterations=100, 
                         mesh_smoothing_factor=0.1):
    """
    Convert a NIfTI segmentation file with multiple segments into separate STL files.
    
    Parameters:
    input_file (str): Path to input .nii.gz file
    output_prefix (str): Prefix for output STL files
    label_map (dict): Optional mapping of segment numbers to output labels
    smoothing_factor (float): Amount of volumetric smoothing to apply
    mesh_smoothing_method (str): 'laplacian', 'taubin', or 'sinc'
    mesh_smoothing_iterations (int): Number of mesh smoothing iterations
    mesh_smoothing_factor (float): Smoothing factor/relaxation factor
    """
    nii_img = nib.load(input_file)
    nii_data = nii_img.get_fdata()
    spacing = nii_img.header.get_zooms()
    
    unique_labels = np.unique(nii_data)
    unique_labels = unique_labels[unique_labels != 0]
    
    for label in unique_labels:
        output_label = label_map.get(int(label), f"segment_{int(label)}") if label_map else f"segment_{int(label)}"
        
        # Volume smoothing
        binary_segment = (nii_data == label)
        smoothed_segment = ndimage.gaussian_filter(binary_segment.astype(float), 
                                                 sigma=smoothing_factor)
        binary_smooth = smoothed_segment > 0.5
        
        # Generate surface mesh
        verts, faces, _, _ = measure.marching_cubes(binary_smooth, spacing=spacing)
        
        # Apply mesh smoothing
        if mesh_smoothing_iterations > 0:
            verts = smooth_mesh_pyvista(verts, faces, 
                                      method=mesh_smoothing_method,
                                      n_iter=mesh_smoothing_iterations,
                                      relaxation_factor=mesh_smoothing_factor)
        
        # Create and save STL
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = verts[face[j]]
        
        output_file = f"{output_prefix}_{output_label}.stl"
        stl_mesh.save(output_file)
        print(f"Saved segment {int(label)} as '{output_label}' to {output_file}")

def convert_to_LPS(vertices):
    vertices[:, 1]= -vertices[:, 1]
    vertices[:, 0] *=-1.0
    return vertices
    


def cap_open_holes(mesh_to_cap):
    """
    Finds and caps large, closed-loop holes in a mesh.
    This is the definitive, robust version that includes:
    1. A pre-repair step to weld topological gaps (merge_vertices).
    2. A safe check (hasattr) to filter out non-closed entities like 'Line'
       objects that were causing crashes.

    Args:
        mesh_to_cap (trimesh.Trimesh): The mesh with open holes.

    Returns:
        trimesh.Trimesh: The capped, watertight mesh.
    """
    print("--- Running Final Robust Planar Hole Capping ---")
    
    mesh = mesh_to_cap.copy()

    # Step 1: Pre-repair the mesh to close topological gaps.
    # This is critical for ensuring visual holes are recognized as closed loops.
    print("Pre-processing: Welding vertices to close topological gaps...")
    mesh.merge_vertices()

    # Step 2: Find all boundary entities.
    hole_loops = mesh.outline()

    caps = []
    for loop_entity in hole_loops.entities:
        # --- THE DEFINITIVE FIX FOR THE ATTRIBUTEERROR ---
        # Use hasattr() to safely check if the entity has the 'is_closed'
        # property. If it doesn't, or if the property is False, we skip it.
        # This correctly handles both 'Line' objects (which lack the property)
        # and any other non-closed loops.
        if not (hasattr(loop_entity, 'is_closed') and loop_entity.is_closed):
            print("Skipping a boundary entity that is not a closed loop.")
            continue
        # --- END OF FIX ---

        # We are now guaranteed to have a valid, closed loop.
        if len(loop_entity.points) < 10:
            print(f"Skipping small closed loop with {len(loop_entity.points)} vertices.")
            continue

        print(f"Found a fillable closed loop with {len(loop_entity.points)} vertices. Attempting to cap.")

        vertices_2d, to_3D_transform = loop_entity.to_planar()
        
        try:
            triangles_2d = trimesh.path.polygons.triangulate_polygon(vertices_2d)
        except Exception as e:
            print(f"Warning: 2D triangulation failed for a hole: {e}. Skipping this hole.")
            continue

        cap_faces = loop_entity.points[triangles_2d]
        cap_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=cap_faces)
        caps.append(cap_mesh)

    if not caps:
        print("No suitable closed-loop holes were found to cap after welding.")
        # Return the pre-repaired mesh even if no holes were capped
        return mesh

    print(f"Stitching {len(caps)} new caps onto the original mesh.")
    final_mesh, _ = trimesh.util.concatenate([mesh] + caps)
    
    final_mesh.merge_vertices()
    final_mesh.fix_normals()

    print("Hole capping complete.")
    return final_mesh

def process_with_parameters(input_file, output_prefix, segment_params, additional_name=None, fill_holes=0, use_pymeshfix = True):
    """
    Process the segmentation with different parameters for each segment.
    
    Parameters:
    segment_params (dict): Dictionary mapping segment labels to parameters:
                         {segment_number: {
                             'smoothing': float,
                             'label': str,
                             'mesh_smoothing_method': str,
                             'mesh_smoothing_iterations': int,
                             'mesh_smoothing_factor': float
                         }}
    """
    nib.openers.Opener.default_compresslevel = 9
    nii_img = nib.load(input_file)
    nii_data = nii_img.get_fdata()
    spacing = nii_img.header.get_zooms()
    
    
    for label, params in segment_params.items():
        volume_smoothing = params.get('smoothing', 0.1)
        output_label = params.get('label', f"segment_{label}")
        mesh_method = params.get('mesh_smoothing_method', 'taubin')
        mesh_iterations = params.get('mesh_smoothing_iterations', 100)
        mesh_factor = params.get('mesh_smoothing_factor', 0.1)
        
        # Volume smoothing
        binary_segment = (nii_data == label)
        if fill_holes > 0:
             binary_segment = fill_holes_3d(binary_segment)
        smoothed_segment = ndimage.gaussian_filter(binary_segment.astype(float), 
                                                 sigma=volume_smoothing)
        binary_smooth = smoothed_segment > 0.5
        
        # Generate surface mesh
        verts, faces, _, _ = measure.marching_cubes(binary_smooth, level=0.5,)
        
        if fill_holes > 0:
            print(f"Analyzing holes for segment {label} ('{output_label}'):")
            hole_stats = analyze_mesh_holes(verts, faces)
            
            # Auto-adjust fill_holes value if needed
            if hole_stats["hole_count"] > 0:
                suggested_fill_size = hole_stats["largest_hole"] * 1.1  # 10% larger than largest hole
                print(f"Suggested fill_holes value: {suggested_fill_size:.2f}")



        #This should fix the shifting issue
        affine = nii_img.affine  # Get the transform matrix
        verts = np.hstack([verts, np.ones((verts.shape[0], 1))])  # Add homogeneous coordinate
        verts = (affine @ verts.T).T[:, :3]  # Apply affine and drop homogeneous coord
        convert_to_LPS(verts)

        if mesh_iterations > 0:
            verts = smooth_mesh_pyvista(verts, faces, 
                                      method=mesh_method,
                                      n_iter=mesh_iterations,
                                      relaxation_factor=mesh_factor)
            
    
        if use_pymeshfix:
            print(f"Attempting Pymeshfix repair")
            try:
                meshfix = pymeshfix.MeshFix(verts, faces)
                meshfix.repair(verbose=False) 
                verts_repaired = meshfix.v
                faces_repaired = meshfix.f
                if faces_repaired.shape[0] > 0:
                    print(f"Pymeshfix repair successful. Original faces: {faces.shape[0]}, Repaired faces: {faces_repaired.shape[0]}")
                    verts, faces = verts_repaired, faces_repaired
                else:
                    print("Pymeshfix resulted in an empty mesh. Using pre-Pymeshfix mesh.")
            except Exception as e:
                print(f"Pymeshfix repair failed: {e}. Using pre-Pymeshfix mesh.")   
            verts = smooth_mesh_pyvista(verts, faces, 
                                        method=mesh_method,
                                        n_iter=100,
                                        relaxation_factor=0.1)
                
        


        # Create and save STL
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = verts[face[j]]
        if "links" in output_prefix:
            output_file = f"{output_prefix}/{output_label + additional_name + '_links'}.stl"
        elif 'rechts' in output_prefix:
                output_file = f"{output_prefix}/{output_label + additional_name + '_rechts'}.stl"
        else: output_file = f"{output_prefix}/{output_label + additional_name}.stl"
        stl_mesh.save(output_file)
        print(f"Saved segment {label} as '{output_label}' to {output_file} with {mesh_iterations} smoothing iterations")


#example of parameters 
segment_params_ftt = {
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
def process_directory( input_dir, output_root_dir,segment_params, fill_holes=0, file_mapping=None, split=False, use_pymeshfix=True):
    for file_name in os.listdir(input_dir):
            if file_name.endswith(".nii.gz"):
                print(file_name)
                input_file = os.path.join(input_dir, file_name)
                # Extract the number from the file name (e.g., "Syn_071.nii.gz" -> "071")
                file_number = file_name.split('_')[1].split('.')[0].split("-")[0]
              
                output_dir = os.path.join(output_root_dir, f"STL{file_number}")
                if split:
                    side = file_name.split('_')[1].split('.')[0].split("-")[1]
                    if side == "links":
                        output_dir+= "_links"
                    elif side == "rechts":
                        output_dir += "_rechts"
                os.makedirs(output_dir, exist_ok=True)
                
                
                print(f"Processing file: {input_file}")
                try:
                    process_with_parameters(input_file, output_dir, segment_params, additional_name=file_number,fill_holes=fill_holes, use_pymeshfix=use_pymeshfix)
                except Exception as e:
                    print(f"Failed to convert {input_file} due to str({e}) ")


def stl_renamer_with_lut(stl_output_path : Path , file_mapping : dict ):
    lookup = {coded["number"]: original_filename for original_filename, coded in file_mapping.items()}
    renamed_dirs=[]
    for item in os.listdir(stl_output_path):
        item_path = os.path.join(stl_output_path, item)
        
      
        if not os.path.isdir(item_path):
            continue
        
        
        if item.startswith('STL') and len(item) >= 6:
            # Extract the number part (3 digits after 'STL')
            number = item[3:6]
            
            # Rest of the directory name (if any)
            rest_of_name = item[6:] if len(item) > 6 else ""
                
            # Check if number exists in lookup
            if number in lookup:
                original_name = lookup[number].replace(".nii.gz", "")
                # Create new directory name (preserving any suffix)
                new_dir_name = f"{original_name}{rest_of_name}"
                new_dir_path = os.path.join(stl_output_path, new_dir_name)
                try: 
                    for file in os.listdir(item_path):
                        old_file_path = os.path.join(item_path, file)
                        if os.path.isfile(old_file_path):
                            new_file_name = original_name + "_" + file
                            new_file_path = os.path.join(item_path, new_file_name)
                            shutil.move(old_file_path, new_file_path)

                except Exception as e:
                    print(f"{str(e)}")
                try:
                    # Check if destination already exists
                    if os.path.exists(new_dir_path):
                        logging.warning(f"Destination {new_dir_path} already exists. Skipping.")
                        continue
                    
                    # Perform the rename operation using shutil.move
                    logging.info(f"Renaming: {item_path} -> {new_dir_path}")
                    shutil.move(item_path, new_dir_path)
                    renamed_dirs.append((item, new_dir_name))
                    
                except Exception as e:
                    logging.error(f"Error renaming {item_path}: {str(e)}")
            else:
                logging.warning(f"No matching original filename found for number {number}")

if __name__ == "__main__":
    process_directory(r"E:\spikes\labels",r"E:\spikes\stl",segment_params=segment_params_ftt)
    