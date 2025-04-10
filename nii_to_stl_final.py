import nibabel as nib
import numpy as np
from skimage import measure
from scipy import ndimage
import pyvista as pv
from stl import mesh
import os 
from scipy.ndimage import binary_fill_holes, binary_closing

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
    # Create faces array in required format for PyVista
    faces_pv = np.hstack([[3, f[0], f[1], f[2]] for f in faces])
    
    # Create PyVista mesh
    mesh_pv = pv.PolyData(vertices, faces_pv)
    
    # Apply smoothing based on method
    if method == 'laplacian':
        mesh_smoothed = mesh_pv.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor)
    elif method == 'taubin':
        mesh_smoothed = mesh_pv.smooth_taubin(n_iter=n_iter, pass_band=relaxation_factor)
    elif method == 'sinc':
        mesh_smoothed = mesh_pv.smooth_sinc(n_iter=n_iter, window_size=relaxation_factor)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return mesh_smoothed.points

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

    return vertices
    
def process_with_parameters(input_file, output_prefix, segment_params, additional_name=None, fill_holes=0):
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
    nii_img = nib.load(input_file)
    nii_data = nii_img.get_fdata()
    spacing = nii_img.header.get_zooms()
    
    
    for label, params in segment_params.items():
        volume_smoothing = params.get('smoothing', 1.0)
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
        verts, faces, _, _ = measure.marching_cubes(binary_smooth, spacing=spacing)
        
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
        
        # Optionally, use the suggested value
        # fill_holes = suggested_fill_size
        # Apply mesh smoothing
        if mesh_iterations > 0:
            verts = smooth_mesh_pyvista(verts, faces, 
                                      method=mesh_method,
                                      n_iter=mesh_iterations,
                                      relaxation_factor=mesh_factor)
            
        if fill_holes > 0:
            print(f"Filling holes up to size {fill_holes} for segment {label}...")
            verts, faces = fill_holes_pyvista(verts, faces, hole_size=fill_holes)
            print(f"Holes filled for segment {label}")
        
        verts=convert_to_LPS(verts)
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



segment_params = {
    1: {
        'label': "Fibula",
        'smoothing': 0.3,  # Less volume smoothing
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 400,  # Fewer iterations
        'mesh_smoothing_factor': 0.1  # Gentler smoothing
    },
    2: {
        'label': "Talus" ,
        'smoothing': 0.9,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 650, # More iterations
        'mesh_smoothing_factor': 0.1  # Stronger smoothing
    
    },
    3: {
        'label': "Tibia",
        'smoothing': 0.8,
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 400,  
        'mesh_smoothing_factor': 0.1
}
}
def process_directory( input_dir, output_root_dir,segment_params, fill_holes=0, file_mapping=None, split=False):
    for file_name in os.listdir(input_dir):
            if file_name.endswith(".nii.gz"):
                print(file_name)
                input_file = os.path.join(input_dir, file_name)
                # Extract the number from the file name (e.g., "Syn_071.nii.gz" -> "071")
                file_number = file_name.split('_')[1].split('.')[0]
              
                output_dir = os.path.join(output_root_dir, f"STL{file_number}")
                if split and file_mapping:
                    for original_name, mapping_data in file_mapping.items():
                        if mapping_data['new_filename'] == file_name.replace(".nii.gz", "_0000.nii.gz"):
                            if 'links' in original_name.lower():
                                output_dir += "_links"
                            elif 'rechts' in original_name.lower():
                                output_dir+= "_rechts"
                            break
                  
                os.makedirs(output_dir, exist_ok=True)
                
                # Call convert_nifti_to_stls for each file
                print(f"Processing file: {input_file}")
                process_with_parameters(input_file, output_dir, segment_params, additional_name=file_number,fill_holes=fill_holes)



shoulder_params={
     1: {
        'label': "Becken",
        'smoothing': 0.1,  # Less volume smoothing
        'mesh_smoothing_method': 'taubin',
        'mesh_smoothing_iterations': 20,  # Fewer iterations
        'mesh_smoothing_factor': 0.005  # Gentler smoothing
    }

}
#process_directory(r"E:\becken\test\NIFTI",r"E:\becken\test\stl",shoulder_params)

