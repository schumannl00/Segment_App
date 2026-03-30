import os
import json
import concurrent.futures
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

def process_single_hu_mask_pair(hu_nii_path, labelmap_path, original_subject_filename, labels_dict, task_id):
    """
    Processes a single image/mask pair and calculates HU statistics.
    """
    results_list = []
    
    try:
        hu_img = nib.load(hu_nii_path)
        mask_img = nib.load(labelmap_path)
        
        # Using get_fdata() is fine, but for large volumes, ensure memory is available
        hu_data = hu_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        if hu_data.shape != mask_data.shape:
            raise ValueError(f"Shape mismatch: HU {hu_data.shape} vs Mask {mask_data.shape}")

        voxel_volume = np.prod(hu_img.header.get_zooms()[:3])
        unique_labels = np.unique(mask_data).astype(int)
        unique_labels = unique_labels[unique_labels != 0]

        for label in unique_labels:
            roi_hu_values = hu_data[mask_data == label]

            if roi_hu_values.size == 0:
                continue

            stats_entry = {
                'Subject_File': original_subject_filename,
                'Label_ID': label,
                'Label_Name': labels_dict.get(str(task_id), {}).get(str(label), 'Unknown'),
                'Voxel_Count': len(roi_hu_values),
                'Voxel_Volume_mm3': len(roi_hu_values) * voxel_volume,
                'Mean_HU': np.mean(roi_hu_values),
                'StdDev_HU': np.std(roi_hu_values),
                'Median_HU': np.median(roi_hu_values),
                'Min_HU': np.min(roi_hu_values),
                'Max_HU': np.max(roi_hu_values),
                'Skewness': stats.skew(roi_hu_values),
                'Kurtosis': stats.kurtosis(roi_hu_values),
                '5th_Percentile_HU': np.percentile(roi_hu_values, 5),
                '95th_Percentile_HU': np.percentile(roi_hu_values, 95),
            }
            results_list.append(stats_entry)
            
    except Exception as e:
        print(f"Error processing {labelmap_path.name}: {e}")
        return []

    return results_list

def calculate_hu_stats(inference_path, labelmap_output_path, file_mapping, output_directory, 
                       max_workers=8, task_id=None, labels_dict=None, stl_metadata_path=None):

    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    
    # Mapping for quick lookup
    num_to_og_map = {info['number']: og_name for og_name, info in file_mapping.items()}
    
    tasks_to_run = []
    
    for labelmap_filename in os.listdir(labelmap_output_path):
        if not (labelmap_filename.endswith(".nii.gz") or labelmap_filename.endswith(".nii")):
            continue

        labelmap_path = Path(labelmap_output_path) / labelmap_filename
        stem = labelmap_path.name.split('.')[0] # Handles .nii.gz better
        
        # Detect Side
        side_suffix = ""
        if "-rechts" in stem:
            side_suffix = "_RECHTS"
            clean_stem = stem.replace("-rechts", "")
        elif "-links" in stem:
            side_suffix = "_LINKS"
            clean_stem = stem.replace("-links", "")
        else:
            clean_stem = stem

        # Extract ID 
        try:
            file_num = clean_stem.split('_')[-1]
            original_subject_base = num_to_og_map.get(file_num)
            
            if not original_subject_base:
                continue

            subject_display_name = Path(original_subject_base).name.split('.')[0] + side_suffix
            
            # Construct HU path (Matching nnUNet naming convention)
            hu_nii_filename = f"{clean_stem}_0000.nii.gz"
            hu_nii_path = Path(inference_path) / hu_nii_filename
            
            if hu_nii_path.exists():
                tasks_to_run.append((
                    hu_nii_path, labelmap_path, subject_display_name, labels_dict, task_id
                ))
            else:
                print(f"Warning: HU file {hu_nii_filename} not found.")
                
        except (IndexError, AttributeError):
            continue

    print(f"Executing {len(tasks_to_run)} tasks...")

    aggregated_results = []
    # ProcessPoolExecutor is generally better for heavy Numpy/Nibabel operations
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_hu_mask_pair, *t) for t in tasks_to_run]
        for future in concurrent.futures.as_completed(futures):
            aggregated_results.extend(future.result())

    if not aggregated_results:
        return False

    df = pd.DataFrame(aggregated_results)

    # STL Merging Logic
    if stl_metadata_path and Path(stl_metadata_path).exists():
        with open(stl_metadata_path, 'r') as f:
            stl_data = json.load(f)
        
        stl_records = []
        for key, metrics in stl_data.items():
            parts = key.split('_')
            stl_records.append({
                'Subject_File': "_".join(parts[:-1]),
                'Label_Name': parts[-1],
                'Mesh_Volume_mm3': metrics.get('Mesh_volume_mm3'),
                'Surface_Area_mm2': metrics.get('Surface_Area_mm2')
            })
        
        stl_df = pd.DataFrame(stl_records)
        df = pd.merge(df, stl_df, on=['Subject_File', 'Label_Name'], how='left')

    # Formatting and Export
    cols = [
        'Subject_File', 'Label_ID', 'Label_Name', 'Voxel_Count', 
        'Voxel_Volume_mm3', 'Mesh_Volume_mm3', 'Surface_Area_mm2',
        'Mean_HU', 'StdDev_HU', 'Median_HU', 'Min_HU', 'Max_HU', 
        'Skewness', 'Kurtosis', '5th_Percentile_HU', '95th_Percentile_HU'
    ]
    
    # Filter only columns that actually exist (in case STL merge failed)
    df = df[[c for c in cols if c in df.columns]]
    df = df.round(4)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_directory / f"HU_Report_{timestamp}.xlsx"
    
    # Excel formatting
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Stats', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Stats']
        header_fmt = workbook.add_format({'bold': True, 'fg_color': '#D7E4BC', 'border': 1})
        
        for i, col in enumerate(df.columns):
            width = max(len(col), df[col].astype(str).map(len).max()) + 2
            worksheet.set_column(i, i, width)
            worksheet.write(0, i, col, header_fmt)

    print(f"Success: {output_path}")
    return True