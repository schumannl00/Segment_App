import os
import shutil
import re
import pydicom
import dicom2nifti
from pathlib import Path
import concurrent.futures
import time # For potential timing/debugging
from dicom2nifti import settings
from pydicom.errors import InvalidDicomError


settings. disable_validate_slice_increment()

def build_pattern(indicator):
    pattern = re.escape(indicator)
    pattern = re.sub(r'\\,', r'\\s*,\\s*', pattern)  # Handle commas
    return re.compile(r"(?<!\w)" + pattern + r"(?!\w)", re.IGNORECASE)


def DICOM_splitter(path):
    """Sorts DICOM files from a source directory into subdirectories
       based on PatientID, Modality, and SeriesDescription."""
    p = Path(path)
   
    sort_dir = p.parent / 'sortiert'
    nifti_out_dir = p.parent / 'NIFTI'

    sort_dir.mkdir(exist_ok=True)
    nifti_out_dir.mkdir(exist_ok=True)

    print(f"Sorting DICOMs from: {p}")
    print(f"Output sorted DICOMs to: {sort_dir}")
    print(f"Output NIFTI files to: {nifti_out_dir}")


    copied_files = 0
    skipped_files = 0
    error_files = 0

    for f in os.listdir(p):
        original_file_path = p / f
        if original_file_path.is_file():
            try:
                read_file = pydicom.dcmread(original_file_path, stop_before_pixels=True, force=True) # Read only header info initially
                file_name = original_file_path.name

                # Determine Patient ID (handle missing attribute)
                if hasattr(read_file, 'PatientID') and read_file.PatientID:
                    file_series_id = str(read_file.PatientID).strip()
                elif hasattr(read_file, 'PatientName') and read_file.PatientName:
                    # Use PatientName as fallback, sanitize it for path usage
                    file_series_id = re.sub(r'[\\/*?:"<>|_]', '-', str(read_file.PatientName).strip())
                else:
                    file_series_id = "UnknownPatient" # Fallback if neither exists

                # Determine Modality (handle missing attribute)
                file_modality = str(getattr(read_file, 'Modality', 'UnknownModality')).strip()
                file_PatientName = re.sub(r'[\\/*?:"<>|_]', '-', str(read_file.PatientName))   #removed  str(read_file.PatientName).strip()
                # Determine Series Description (handle missing attribute)
                file_series_description = str(getattr(read_file, 'SeriesDescription', 'UnknownSeries')).strip()
                # Sanitize description for path usage
                file_series_description = re.sub(r'[\\/*?:"<>|]', '_', file_series_description)
                if not file_series_description:
                    file_series_description = "UnknownSeries"

                
                target_dir_name = f"{file_series_id}_{file_PatientName}_{file_series_description}"  
                description_path = sort_dir / target_dir_name

                
                description_path.mkdir(exist_ok=True)

                # Copy the file
                target_file_path = description_path / file_name
                # Avoid re-copying if file exists (optional, can speed up re-runs)
                if not target_file_path.exists():
                     shutil.copyfile(original_file_path, target_file_path)
                     copied_files += 1
                else:
                     skipped_files +=1


            except Exception as e:
                print(f"Error processing file {original_file_path}: {e}")
                error_files += 1
                continue 

    print(f"DICOM Sorting Summary:")
    print(f"  Copied: {copied_files}")
    print(f"  Skipped (already exist): {skipped_files}")
    print(f"  Errors: {error_files}")
    return sort_dir, nifti_out_dir 

# --- Helper function for converting a single series ---
def convert_single_series_to_nifti(input_dir_path, output_nifti_path):
    
    """Converts a single directory of DICOM series to a NIFTI file."""
    try:
        print(f"Attempting conversion: {input_dir_path} -> {output_nifti_path}")
        
        Path(output_nifti_path).parent.mkdir(exist_ok=True)
        dicom2nifti.dicom_series_to_nifti(str(input_dir_path), str(output_nifti_path), reorient_nifti=True) 
        print(f"Successfully converted: {output_nifti_path.name}")
        return (str(input_dir_path), True, str(output_nifti_path)) 
    except Exception as e:
        # Catch specific dicom2nifti errors if possible, otherwise general Exception
        error_msg = f"Failed conversion for {input_dir_path.name}: {e}"
        print(error_msg)
        # Optionally try to remove partially created NIFTI file if conversion fails
        if output_nifti_path.exists():
            try:
                os.remove(output_nifti_path)
                print(f"Removed potentially incomplete file: {output_nifti_path.name}")
            except OSError as remove_error:
                print(f"Could not remove incomplete file {output_nifti_path.name}: {remove_error}")
        return (str(input_dir_path), False, error_msg) # Return input, failure status, error message

# --- Main function using ProcessPoolExecutor ---
def raw_data_to_nifti_parallel(raw_path, scans_indicators=None, use_default=False, max_workers=12):
    """
    Sorts DICOMs and converts selected series to NIFTI in parallel.

    Args:
        raw_path (str): Path to the directory containing raw DICOM files.
        scans_indicators (list, optional): List of strings. Only series whose
                                           directory name contains one of these
                                           indicators will be converted (unless use_default=True).
                                           Defaults to None.
        use_default (bool, optional): If True, attempts to convert all series,
                                      ignoring scans_indicators. Defaults to False.
        max_workers (int, optional): Maximum number of processes to use for conversion.
                                     Defaults to 12.
    """
    start_time = time.time()
    p = Path(raw_path)
    if not p.is_dir():
        print(f"Error: Raw path '{raw_path}' not found or is not a directory.")
        return

    # 1. Sort DICOMs sequentially first
    print("-" * 20)
    print("Step 1: Sorting DICOM files...")
    try:
        sort_dir, nifti_out_dir = DICOM_splitter(raw_path)
        if not sort_dir or not nifti_out_dir:
             print("DICOM splitting failed to return valid directories.")
             return
    except Exception as e:
        print(f"Error during DICOM splitting: {e}")
        return
    print("Step 1: Sorting complete.")
    print("-" * 20)


    # 2. Prepare list of conversion tasks
    print("Step 2: Preparing conversion tasks...")
    tasks = []
    patterns = []
    if scans_indicators and not use_default:
        # Compile regex patterns for faster matching if indicators are provided
        try:
            patterns = [build_pattern(indicator) for indicator in scans_indicators]
            print(f"Using scan indicators: {scans_indicators}")
        except re.error as e:
            print(f"Error compiling regex for scan indicators: {e}. Please check indicators: {scans_indicators}")
            return 

    skipped_dirs = []
    dirs_to_convert = []

    for item_name in os.listdir(sort_dir):
        full_path = sort_dir / item_name
        if full_path.is_dir():
    
            should_convert = False
            if use_default:
                should_convert = True
            elif patterns: 
                if len(item_name.split("_"))>3:
                    name= item_name.split("_")[2]
                    for i in range(3,len(item_name.split("_"))):
                            name += "_"+ item_name.split("_")[i]
                    if any(pattern.search(name) for pattern in patterns):
                        should_convert = True
                    else:  skipped_dirs.append(item_name)
                elif any(pattern.search(item_name.split("_")[2]) for pattern in patterns):
                     should_convert = True
                else:
                     skipped_dirs.append(item_name)
            else: 
                 skipped_dirs.append(item_name)


            if should_convert:
                nifti_filename = f"{item_name}.nii.gz"
                nifti_path = nifti_out_dir / nifti_filename
                
                tasks.append((full_path, nifti_path))
                dirs_to_convert.append(item_name)

    if not tasks:
        print("No series found matching the criteria for conversion.")
        if skipped_dirs:
             print(f"Skipped directories based on indicators: {len(skipped_dirs)}")
        return 

    print(f"Found {len(tasks)} series to convert:")
    #for name in dirs_to_convert: print(f"  - {name}") # Uncomment for verbose listing
    if skipped_dirs:
        print(f"Skipped {len(skipped_dirs)} directories based on indicators.")
        #for name in skipped_dirs: print(f"  - {name}") # Uncomment for verbose listing

    print("Step 2: Preparation complete.")
    print("-" * 20)

    # 3. Execute conversions in parallel
    print(f"Step 3: Starting parallel conversion using up to {max_workers} processes...")
    results = []
    successful_conversions = 0
    failed_conversions = 0

    # Use ProcessPoolExecutor
    # The 'with' statement ensures the pool is properly shut down
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks. submit returns a Future object.
        future_to_input = {executor.submit(convert_single_series_to_nifti, task[0], task[1]): task[0] for task in tasks}

        
        for future in concurrent.futures.as_completed(future_to_input):
            input_dir = future_to_input[future]
            try:
                result = future.result() # Get result (input_path, success_bool, output_path/error_msg)
                results.append(result)
                if result[1]:
                    successful_conversions += 1
                else:
                    failed_conversions += 1
            except Exception as exc:
                
                error_msg = f"Task for {input_dir.name} generated an exception: {exc}"
                print(error_msg)
                results.append((str(input_dir), False, error_msg))
                failed_conversions += 1

    print("Step 3: Parallel conversion complete.")
    print("-" * 20)

    # 4. Report Summary
    end_time = time.time()
    print("Conversion Summary:")
    print(f"  Total series attempted: {len(tasks)}")
    print(f"  Successful conversions: {successful_conversions}")
    print(f"  Failed conversions: {failed_conversions}")
    if failed_conversions > 0:
        print("  Failures:")
        for res in results:
            if not res[1]: # If conversion failed
                print(f"    - {Path(res[0]).name}: {res[2]}") # Print input dir name and error message
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("-" * 20)

    # You might want to return the results list for detailed logging or GUI feedback
    return results

# Example Usage:
# Assuming your raw DICOMs are in 'C:/dicom_raw_data'
# And you want to convert series containing 'T1' or 'T2'
# raw_data_to_nifti_parallel('C:/dicom_raw_data', scans_indicators=['T1', 'T2'], max_workers=12)

# To convert everything (use with caution, might include localizers, dose reports etc.):
# raw_data_to_nifti_parallel('C:/dicom_raw_data', use_default=True, max_workers=12)

# Blueprint for changing metadata of DICOM files, here just series description 
# For more Keywords, tags see https://dicom.innolitics.com/ciods/rt-dose/patient/00100010
def modify_metadata(dcm_file, backup=True, new_desc= '', new_name = ''):
    try:
        ds = pydicom.dcmread(dcm_file)
        
        # Create backup if requested
        if backup:
            ds.save_as(dcm_file.replace('.dcm', '_backup.dcm'))
        
        # Get acquisition number (default to empty string if not present)
        acq_num = str(ds.get('AcquisitionNumber', ''))
        study_id= (0x0020,0x0010)  #tags as names are unreliable
        series_tag=(0x0021,0x1003) #tags as names are unreliable
        patient_study_id= ds[study_id].value
        patient_series= ds[series_tag].value
        
        series_desc = str(ds.get('SeriesDescription', '')) # Get current series description (default to empty string if not present)
        patient_name = str(ds.get('	PatientName', ''))   # Get current patient name (default to empty string if not present)
        # Modify series description, example change as needed or change for patient name if desired in the same format (skip if statement if not needed)
        modified = False
        if new_desc:
            print(f"Updating SeriesDescription for: {dcm_file}")
            ds.SeriesDescription = new_desc
            modified = True

        if new_name:
            print(f"Updating PatientName for: {dcm_file}")
            # Corrected attribute from 'Patientname' to 'PatientName' (case-sensitive)
            # Corrected tag lookup typo from '\tPatientName' to 'PatientName'
            ds.PatientName = new_name
            modified = True
            
        # Save the file only if changes were made
        if modified:
            ds.save_as(dcm_file)
            return True
        else:
            print(f"No new name or description provided for {dcm_file}. No changes made.")
            return False
        
    except InvalidDicomError:
        print(f"Error: {dcm_file} is not a valid DICOM file")
        return False
    except PermissionError:
        print(f"Error: No write permissions for {dcm_file}")
        return False
    except Exception as e:
        print(f"Unexpected error processing {dcm_file}: {str(e)}")
        return False



# Modifies all DICOMs in the given directory with the specific instructions given in modify_series_description so modify than accordingly especially the new_desc variable 

def modify_dcms(directory_path, new_desc, new_name):
    path= Path(directory_path)
    mod_files = 0 
    for root, dirs, files in os.walk(path):
        print(root)
        for file in files:
            file_path= os.path.join(root, file)
            if os.path.isfile(file_path):
                modify_metadata(file_path, backup=False, new_desc=new_desc, new_name=new_name) 
                mod_files += 1
    print(mod_files)

#Just a simple renamer function for the nnUNet convention without any LUT 
def nifti_renamer(nii_path, prefix, suffix, number, file_mapping):
    p=Path(nii_path)
    dir= p.parent
    filled_number = str(number+1).zfill(3)
    newpath= os.path.join(dir,f"{prefix}"+filled_number+ f'{suffix}.nii.gz')
    os.rename(nii_path, newpath)
    file_mapping[os.path.basename(nii_path)] = {
        "new_filename": os.path.basename(newpath),
        "number": filled_number
    }


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # This is needed for Windows
    # Your function call here
    raw_data_to_nifti_parallel(r"E:\test5\00002168", use_default=True, max_workers=12)   
