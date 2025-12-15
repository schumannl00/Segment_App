import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats # For Skewness and Kurtosis
from pathlib import Path
import os
import concurrent.futures # For parallel execution
import json

# Define the standard nnUNet single-channel suffix (as determined from context)


# Helper function to process a single NIfTI image and mask pair
def process_single_hu_mask_pair(hu_nii_path : Path , labelmap_path: Path , original_subject_filename : str, labelmap_filename : str , hu_nii_filename: str, id: int, labels_dict: dict) -> list:
    """
    Processes a single image/mask pair, calculates HU statistics, merges mesh data,
    and computes T-scores for specified bone density labels.
    """ 


    
    results_list = []
    
    try:
        # Load NIfTI Image and Mask
        print(f"Processing: {hu_nii_filename} and {labelmap_filename}")
        hu_img = nib.load(hu_nii_path)
        mask_img = nib.load(labelmap_path)
        
        hu_data = hu_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Voxel size for volume calculation
        voxel_volume = np.prod(hu_img.header.get_zooms()[:3]) 
        
        if hu_data.shape != mask_data.shape:
            # This should have been caught during nnUNet prediction, but check for safety
            raise ValueError(f"Shape mismatch: HU ({hu_data.shape}) vs Mask ({mask_data.shape})")

        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels != 0] # Exclude background (0)

        # Extract the simplified nnUNet base name from the labelmap filename
        nnunet_simple_name = Path(labelmap_filename).stem.split('.')[0]

        for label in unique_labels:
            boolean_mask = (mask_data == label)
            roi_hu_values = hu_data[boolean_mask]

            if roi_hu_values.size == 0:
                continue 

            # --- 1. Fast Voxel-based Statistics ---
            
            mean_hu = np.mean(roi_hu_values)
            std_hu = np.std(roi_hu_values)
            
            skewness = stats.skew(roi_hu_values)
            kurtosis = stats.kurtosis(roi_hu_values)


            stats_entry = {
                'Subject_File': original_subject_filename, 
                'Label_ID': int(label),
                'Label_Name': f"{labels_dict.get(id, {}).get(str(int(label)), 'Unknown')}", 
                'Voxel_Count': len(roi_hu_values),
                'Voxel_Volume_mm3': len(roi_hu_values) * voxel_volume, 
                'Mean_HU': mean_hu,
                'StdDev_HU': std_hu,
                'Median_HU': np.median(roi_hu_values),
                'Min_HU': np.min(roi_hu_values),
                'Max_HU': np.max(roi_hu_values),
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                '25th_Percentile_HU': np.percentile(roi_hu_values, 25),
                '75th_Percentile_HU': np.percentile(roi_hu_values, 75), } 
                
            
            results_list.append(stats_entry)
            
    except Exception as e:
        print(f"Error processing pair {labelmap_filename}: {str(e)}")
        # Return an empty list on failure for aggregation
        return [] 

    return results_list

# The main function using concurrent futures
def calculate_hu_stats(inference_path : Path, labelmap_output_path : Path, file_mapping : dict, output_directory : Path , max_workers :int =12, id : str = None, labels_dict : dict =None):

    

    NNUNET_CHANNEL_SUFFIX = "_0000"
 

   
    # Create inverted mapping: nnUNet_Base_Name (e.g., 'Leg_001_0000') -> Original_Nii_Filename (e.g., 'PID_PNAME_...')
    inverted_mapping = {}
    for original_nii_name, nnunet_info in file_mapping.items():
        nnunet_base_name = Path(nnunet_info['new_filename']).stem.split('.')[0] 
        inverted_mapping[nnunet_base_name] = original_nii_name
    
    # -------------------------------
    
    tasks_to_run = []
    
    # 1. Prepare all tasks (collect paths and parameters)
    for labelmap_filename in os.listdir(labelmap_output_path):
        if labelmap_filename.endswith(".nii.gz") or labelmap_filename.endswith(".nii"):
            
            labelmap_base_name = Path(labelmap_filename).stem.split('.')[0] 
            
            original_subject_filename = inverted_mapping.get(labelmap_base_name + NNUNET_CHANNEL_SUFFIX)
            if not original_subject_filename: continue

            new_file_info = file_mapping.get(original_subject_filename)
            if not new_file_info: continue
                 
            nnunet_simple_name = Path(new_file_info['new_filename']).stem.split('.')[0]
            
            # Construct the HU filename by appending the channel suffix
            hu_nii_filename = f"{nnunet_simple_name}.nii.gz"
            
            hu_nii_path = Path(inference_path) / hu_nii_filename
            labelmap_path = Path(labelmap_output_path) / labelmap_filename
            
            if not hu_nii_path.exists(): 
                print(f"Warning: Missing HU file {hu_nii_filename} for mask {labelmap_filename}. Skipping.")
                continue
            
            # Append task parameters
            tasks_to_run.append((hu_nii_path, labelmap_path, original_subject_filename, labelmap_filename, hu_nii_filename, id, labels_dict))

    print(f"Prepared {len(tasks_to_run)} pairs for parallel HU statistics calculation.")

    # 2. Execute in parallel using ThreadPoolExecutor
    aggregated_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_hu_mask_pair, *task): task[2] for task in tasks_to_run
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            original_name = future_to_task[future]
            try:
                results_from_task = future.result() 
                aggregated_results.extend(results_from_task)
            except Exception as exc:
                print(f"Parallel task for {original_name} failed: {exc}")


    if not aggregated_results:
        print("No statistics calculated. Check file paths and data integrity.")
        return False

    # 3. Create DataFrame and export
    df = pd.DataFrame(aggregated_results)
    
    # Define a consistent column order (Final structure)
    column_order = [
        'Subject_File',  'Label_ID', 'Label_Name', 
        'Voxel_Count', 'Voxel_Volume_mm3', 
        'Mean_HU', 'StdDev_HU', 'Median_HU', 'Min_HU', 'Max_HU', 
        'Skewness', 'Kurtosis', 
        '25th_Percentile_HU', '75th_Percentile_HU', 
    
    ]
    df = df[column_order]

    df_formatted = df.round({
    'Voxel_Volume_mm3': 3,  # Example: 3 decimal places for volume
    'Mean_HU': 4,
    'StdDev_HU': 4,
    'Median_HU': 4,
    'Skewness': 4,
    'Kurtosis': 4,
    '25th_Percentile_HU': 4,
    '75th_Percentile_HU': 4,
    })
    
    output_filename = f"HU_Statistics_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    Path(output_directory).mkdir(exist_ok=True, parents=True)

    csv_path = Path(output_directory) / f"{output_filename}.csv"
    df_formatted.to_csv(csv_path, index=False)
    print(f"HU statistics saved to: {csv_path}")

    excel_path = Path(output_directory) / f"{output_filename}.xlsx"

    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_formatted.to_excel(writer, sheet_name='Sheet1', index=False)

      
        workbook  = writer.book
        worksheet = writer.sheets['Sheet1']

        
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color':"#B7FFEF" , 
            'border': 1
        })

        # Iterate through each column to find the max width
        for i, column in enumerate(df_formatted.columns):
            # Calculate width of column header
            column_len = len(str(column))

            # Calculate max width of data in that column
            # (We use map(str) to handle non-string data)
            max_data_len = df_formatted[column].map(str).map(len).max()

            # Set the width to the larger of the two, plus a little padding
            # If max_data_len is NaN (empty column), default to header length
            if pd.isna(max_data_len):
                max_data_len = 0
                
            adjusted_width = max(column_len, max_data_len) + 2
            
            # Apply the width and the header format
            worksheet.set_column(i, i, adjusted_width)
            worksheet.write(0, i, column, header_format)

    print(f"HU statistics saved to: {excel_path}")
    return True

if __name__ == "__main__":
    # Example usage (paths would need to be set appropriately)
    inference_dir = Path("C:\\Users\\schum\\Desktop\\zesbo\\label_ver\\nii")
    labelmap_dir = Path("C:\\Users\\schum\\Desktop\\zesbo\\label_ver\\label")
    output_dir = Path("C:\\Users\\schum\\Desktop\\zesbo\\label_ver\\output_stats")
    
    # Example file mapping
    example_file_mapping = {
      "Scan_01" : {"new_filename": "Sco_001.nii.gz", "number" : "001"},
        "Scan_02" : {"new_filename": "Sco_002.nii.gz", "number" : "002"},
        "Scan_03" : {"new_filename": "Sco_003.nii.gz", "number" : "003"},
          "Scan_04" : {"new_filename": "Sco_004.nii.gz", "number" : "004"}
    }
    labels_dict = { "117": {
        "1" : "C",
        "2" : "T",
        "3" : "L", 
        "4" : "Ribs"
    }}
    
    calculate_hu_stats(inference_dir, labelmap_dir, example_file_mapping, output_dir, id="117", labels_dict=labels_dict, stl_metadata_path=None)