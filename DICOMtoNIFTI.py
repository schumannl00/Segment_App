import os
import shutil
import re
import pydicom
import dicom2nifti
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import time # For potential timing/debugging
from dicom2nifti import settings
from pydicom.errors import InvalidDicomError
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

settings. disable_validate_slice_increment()
#rewrote  so no splitting needed anymore, any spaces are replaced by - already 
def build_pattern(indicator):
    forbidden_boundary = r"[\w.+\-]"
    escaped_indicator = re.escape(indicator)
    pattern = f"(?<!{forbidden_boundary}){escaped_indicator}(?!{forbidden_boundary})"
    return re.compile(pattern, re.IGNORECASE)




# Precompiled regex
sanitize_general = re.compile(r'[\\/*?:"<>|_]')
sanitize_spaces = re.compile(r'\s+')

def clean_string(s):
    s = sanitize_general.sub("-", s)
    s = sanitize_spaces.sub("-", s)
    s = s.strip().rstrip(".")
    return s or "Unknown"


def safe_copy(src_dst):
    src, dst = src_dst
    try:
        shutil.copyfile(src, dst)
        return True
    except Exception as e:
        return e


def DICOM_splitter(path, max_workers=32, use_only_name : bool = True):
    p = Path(path)

    sort_dir = p.parent / 'sortiert'
    nifti_out_dir = p.parent / 'NIFTI'
    sort_dir.mkdir(exist_ok=True)
    nifti_out_dir.mkdir(exist_ok=True)

    print(f"Sorting DICOMs from: {p}")
    print(f"Output sorted DICOMs to: {sort_dir}")
    print(f"Output NIFTI files to: {nifti_out_dir}")
    print(f"Using only PatientName for folder naming: {use_only_name}")

    copied_files = 0
    skipped_files = 0
    error_files = 0

    needed_tags = [
        "PatientID",
        "PatientName",
        "Modality",
        "SeriesDescription",
        "SeriesNumber"]

    files_to_copy = []
    known_dirs = set()  # avoid repeated mkdir costs

    # -----------------------------
    # PASS 1: FAST SCAN + PARSE
    # -----------------------------
    print("Scanning DICOM files and preparing copy list...")
    for original_file_path in p.rglob("*"):
        if not original_file_path.is_file():
            continue

        try:
            dcm = pydicom.dcmread(
                original_file_path,
                stop_before_pixels=True,
                force=True,
                specific_tags=needed_tags,
            )

            # Extract tags
            pid = getattr(dcm, "PatientID", "UnknownID")
            pname = getattr(dcm, "PatientName", "UnknownName")
            sdesc = getattr(dcm, "SeriesDescription", "UnknownSeries")
            snum = getattr(dcm, "SeriesNumber", 0)

            pid = clean_string(str(pid))
            pname = clean_string(str(pname))
            sdesc = clean_string(str(sdesc))
            snum = str(snum)

            if use_only_name:
                folder = sort_dir / f"{pname}_{pname}_Series{snum}_{sdesc}"
            else:
                folder = sort_dir / f"{pid}_{pname}_Series{snum}_{sdesc}"

            # Create folder once
            if folder not in known_dirs:
                folder.mkdir(exist_ok=True, parents=True)
                known_dirs.add(folder)

            target = folder / original_file_path.name

            if not target.exists():
                files_to_copy.append((original_file_path, target))
            else:
                skipped_files += 1

        except Exception as e:
            print(f"Failed reading {original_file_path}: {e}")
            error_files += 1
    
    # ------------------------------------------------------
    # PASS 2: PARALLEL COPY (I/O bound → threads are ideal)
    # ------------------------------------------------------
    print(f"\nStarting parallel copy of {len(files_to_copy)} files...")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(safe_copy, pair) for pair in files_to_copy]

        for fut in as_completed(futures):
            result = fut.result()
            if result is True:
                copied_files += 1
            else:
                error_files += 1

    # -----------------------------
    # Summary
    # -----------------------------
    print("\nDICOM Sorting Summary:")
    print(f"  Copied: {copied_files}")
    print(f"  Skipped: {skipped_files}")
    print(f"  Errors: {error_files}")

    return sort_dir, nifti_out_dir


# --- Helper function for converting a single series ---
def convert_single_series_to_nifti(input_dir_path, output_nifti_path):
    
    """Converts a single directory of DICOM series to a NIFTI file."""
    try:
        print(f"Attempting conversion: {input_dir_path} -> {output_nifti_path}")
        
        Path(output_nifti_path).parent.mkdir(exist_ok=True)
        dicom2nifti.dicom_series_to_nifti(str(input_dir_path), str(output_nifti_path), reorient_nifti=True )

        nii = nib.load(str(output_nifti_path)) 
        orig_orient = io_orientation(nii.affine)
        target_orient = axcodes2ornt(('R', 'A', 'S'))
        if not np.array_equal(orig_orient, target_orient):
            transform = ornt_transform(orig_orient, target_orient)
            nii_ras = nii.as_reoriented(transform)
            nib.save(nii_ras, str(output_nifti_path))
            print(f"Reoriented NIFTI to RAS for: {output_nifti_path.name}")

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
def raw_data_to_nifti_parallel(raw_path, scans_indicators=None, group_filter = None, use_default=False, max_workers=12, use_only_name=True, max_workers_dicom=32):
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
        use_only_name (bool, optional): If True, uses only PatientName for folder naming during DICOM splitting.
                                        Defaults to True.
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
        sort_dir, nifti_out_dir = DICOM_splitter(raw_path, max_workers=max_workers_dicom, use_only_name=use_only_name)
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
    group_pattern = []
    if scans_indicators and not use_default:
        # Compile regex patterns for faster matching if indicators are provided
        try:
            patterns = [build_pattern(clean_string(indicator)) for indicator in scans_indicators]
            print(f"Using scan indicators: {scans_indicators}")
        except re.error as e:
            print(f"Error compiling regex for scan indicators: {e}. Please check indicators: {scans_indicators}")
            return 
    if group_filter and not use_default:
        try:
            group_pattern = re.compile(re.escape(group_filter), re.IGNORECASE)
            print(f"Using group filter: '{group_filter}'")
        except re.error as e:
            print(f"Error compiling regex for group filter: {e}. Please check filter: {group_filter}")
            return

    skipped_dirs = []
    dirs_to_convert = []

    for item_name in os.listdir(sort_dir):
        full_path = sort_dir / item_name
        if full_path.is_dir():
            parts = item_name.split("_")
            should_convert = False
            if use_default:
                should_convert = True
            else: 
                match_indicator = False
                match_group = False

                if patterns:
                    if len(parts) >= 4:
                        series_description_from_name = "_".join(parts[3:])  # The description is everything from the 4th element onwards, joined by underscores
                        if any(p.search(series_description_from_name) for p in patterns):
                            match_indicator = True
                    elif len(parts) < 4:
                        print(f"Unexpected name format {item_name} for indicator check")
                
                
                if group_pattern:
                    if group_pattern.search(item_name):
                        match_group = True
                
                # Convert if *either* filter matches (and filters are active)
                if (patterns and match_indicator) or (group_pattern and match_group):
                    should_convert = True

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





def modify_metadata(dcm_file, backup=True, custom = False, new_desc= '', new_name = ''):
    # Blueprint for changing metadata of DICOM files, here just series description 
    # For more Keywords, tags see https://dicom.innolitics.com/ciods/rt-dose/patient/00100010
    try:
        ds = pydicom.dcmread(dcm_file)
        
        # Create backup if requested
        if backup:
            ds.save_as(dcm_file.replace('.dcm', '_backup.dcm'))
        
        # Get acquisition number (default to empty string if not present)
        acq_num = str(ds.get('AcquisitionNumber', ''))
        #study_id= (0x0020,0x0010)  #tags as names are unreliable 
        #series_tag=(0x0021,0x1003) #tags as names are unreliable, check if the tag is there before uncommenting, it is private in most cases
        #patient_study_id= ds[study_id].value
        #patient_series= ds[series_tag].value
        
        series_desc = str(ds.get('SeriesDescription', '')) # Get current series description (default to empty string if not present)
        patient_name = str(ds.get('PatientName', ''))   # Get current patient name (default to empty string if not present)
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
        
        if custom: 
            #play around with what you need here
            print("Custom changes made.")
            series_number_tag = (0x0020,0x0011)
            series_number = ds[series_number_tag].value
            string_number = str(series_number) 
            print(series_number)
            ds.SeriesDescription = f"{series_desc}_{string_number}" 
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

def modify_dcms(directory_path, new_desc, new_name,custom):
    path= Path(directory_path)
    mod_files = 0 
    for root, dirs, files in os.walk(path):
        print(root)
        for file in files:
            file_path= os.path.join(root, file)
            if os.path.isfile(file_path):
                modify_metadata(file_path, backup=False, new_desc=new_desc, new_name=new_name, custom=custom) 
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
    raw_data_to_nifti_parallel(r"C:\Users\schum\Downloads\Covid Scans\Covid Scans\Subject (1)\test", use_default=False, max_workers=8, use_only_name=False, scans_indicators=["Lung 1.5",])