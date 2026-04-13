import os
import nibabel as nib
import numpy as np
import scipy.ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed

def resample_nifti(img, target_spacing, order=3):
    """
    Resample a NIfTI image to target spacing and update the affine scaling.
    """
    data = img.get_fdata()
    affine = img.affine
    original_spacing = img.header.get_zooms()[:3]
    
    # Calculate zoom factors based on x, y, z
    zoom_factors = [s / t for s, t in zip(original_spacing, target_spacing)]
    
    # Perform the zoom
    resampled_data = scipy.ndimage.zoom(data, zoom=zoom_factors, order=order)
    
    # Casting logic: order 0 (labels) stays int16, images stay original dtype
    if order == 0:
        resampled_data = np.round(resampled_data).astype(np.int16)
    else:
        resampled_data = resampled_data.astype(img.get_data_dtype())

    # Update affine scaling while preserving rotation and origin
    new_affine = affine.copy()
    for i in range(3):
        direction = affine[:3, i] / np.linalg.norm(affine[:3, i])
        new_affine[:3, i] = direction * target_spacing[i]

    return nib.Nifti1Image(resampled_data, new_affine)

def process_file_worker(filename, input_dir, output_dir, target_spacing, is_label):
    """ Worker task: Loads from input_dir, saves to output_dir """
    try:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        order = 0 if is_label else 3
        
        img = nib.load(in_path)
        resampled_img = resample_nifti(img, target_spacing, order=order)
        
        nib.save(resampled_img, out_path)
        return f"✅ {filename}"
    except Exception as e:
        return f"❌ {filename}: {str(e)}"

def main():
    # --- CONFIGURATION ---
    TARGET_SPACING = (1.0, 1.0, 1.0)  # (x, y, z) in mm
    MAX_WORKERS = 14
    
    # Define your source and destination pairs
    # Logic: (Source_Folder, Destination_Folder, Is_Label_Flag)
    tasks = [
       
        (
            r"D:\nnUNet_raw\Dataset217_Spine\labels_oldspacing", 
            r"D:\nnUNet_raw\Dataset217_Spine\labelsTr", 
            True
        )
    ]
    # ---------------------

    for src_dir, dst_dir, is_label in tasks:
        if not os.path.exists(src_dir):
            print(f"Skipping: {src_dir} not found.")
            continue
            
        os.makedirs(dst_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.endswith(".nii.gz")]
        
        print(f"\nProcessing {len(files)} files: {src_dir} -> {dst_dir}")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_file_worker, f, src_dir, dst_dir, TARGET_SPACING, is_label) 
                for f in files
            ]
            
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                res = future.result()
                if "❌" in res:
                    print(res)
                if done_count % 10 == 0 or done_count == len(files):
                    print(f"Progress: {done_count}/{len(files)}")

    print("\nResampling finished successfully.")

if __name__ == "__main__":
    main()