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
from typing import List, Pattern, Tuple, Set, Dict, Optional 
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, DirectoryPath, FilePath, field_validator, ConfigDict
from abc import ABC, abstractmethod


settings.disable_validate_slice_increment()

# Precompiled regex, reduces overhead 
sanitize_general : re.Pattern = re.compile(r'[\\/*?:"<>|_]')
sanitize_spaces : re.Pattern = re.compile(r'\s+')

def build_pattern(indicator : str ) -> re.Pattern:
    forbidden_boundary = r"[\w.+\-]"
    escaped_indicator = re.escape(indicator)
    pattern = f"(?<!{forbidden_boundary}){escaped_indicator}(?!{forbidden_boundary})"
    return re.compile(pattern, re.IGNORECASE)

def clean_string(s : str ) -> str:
    s = sanitize_general.sub("-", s)
    s = sanitize_spaces.sub("-", s)
    s = s.strip().rstrip(".")
    return s or "Unknown"

# use as fallback if needed
def safe_copy(src_dst : Tuple[str, str]):
    src, dst = src_dst
    try:
        shutil.copyfile(src, dst)
        return True
    except Exception as e:
        return e
    
    
def safe_link(src_dst : Tuple[str, str]):  
    src, dst = src_dst 
    try:
        src_abs = os.path.abspath(src)
        dst_abs = os.path.abspath(dst)
        
        # Windows Long Path Support: Use the \\?\ prefix if on Windows
        if os.name == 'nt' and not src_abs.startswith('\\\\?\\'):
            src_abs = '\\\\?\\' + src_abs
            dst_abs = '\\\\?\\' + dst_abs

        # Use hard links instead of symlinks to avoid Admin permission issues
        os.link(src_abs, dst_abs)
        return True
    except FileExistsError:
        return True 
    except Exception as e:
        print(f"Link Error: {e}")
        return e


class NiftiConfig(BaseModel):
    raw_path: DirectoryPath
    scans_indicators: Optional[List[str]] = None
    group_filter: Optional[str] = None
    use_default: bool = True
    max_workers: int = Field(default=14, gt=0)
    use_only_name: bool = True
    max_workers_dicom: int = Field(default=32, gt=0)

    # Automatically compile indicators into patterns if they exist
    def get_patterns(self) -> List[re.Pattern]:
        if not self.scans_indicators:
            return []
        return [build_pattern(clean_string(ind)) for ind in self.scans_indicators]
    
    @field_validator('raw_path', mode='before')
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if isinstance(v, str) else v

class ConversionTask(BaseModel):
    input_dir: DirectoryPath
    output_path: Path # Not yet a FilePath because it doesn't exist yet
    model_config=ConfigDict(arbitrary_types_allowed = True)
        
        
class BaseConverter(ABC):
    def __init__(self, config: NiftiConfig):
        self.config = config

    @abstractmethod
    def run(self):
        """Orchestrates the sorting and parallel conversion."""
        pass

    @staticmethod
    @abstractmethod
    def process_item(task: ConversionTask) -> Tuple[str, bool, str]:
        """The core conversion logic executed in parallel."""
        pass
    
    
class NiftiParallelConverter(BaseConverter):
    def run(self):
        start_time = time.time()
        print(f"{'-'*20}\nStep 1: Sorting DICOM files...")
        
        # 1. Sort DICOMs (utilizing your existing DICOM_splitter function)
        sort_dir, nifti_out_dir = DICOM_splitter(
            self.config.raw_path, 
            max_workers=self.config.max_workers_dicom, 
            use_only_name=self.config.use_only_name
        )
        
        # 2. Prepare validated tasks
        print("Step 2: Preparing conversion tasks...")
        tasks = self._prepare_tasks(sort_dir, nifti_out_dir)
        
        if not tasks:
            print("No series found matching the criteria.")
            return

        # 3. Parallel Execution
        print(f"Step 3: Starting conversion with {self.config.max_workers} processes...")
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {executor.submit(self.process_item, task): task for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    results.append(future.result())
                except Exception as exc:
                    task = future_to_task[future]
                    results.append((str(task.input_dir), False, str(exc)))

        # 4. Summary Reporting
        self._report(results, start_time)

    def _prepare_tasks(self, sort_dir: Path, nifti_out_dir: Path) -> List[ConversionTask]:
        tasks = []
        # Compile patterns once
        patterns = [build_pattern(clean_string(ind)) for ind in (self.config.scans_indicators or [])]
        group_pat = re.compile(re.escape(self.config.group_filter), re.IGNORECASE) if self.config.group_filter else None

        for item_name in os.listdir(sort_dir):
            full_path = sort_dir / item_name
            if not full_path.is_dir(): continue
            
            should_convert = self.config.use_default
            if not should_convert:
                parts = item_name.split("_")
                series_desc = "_".join(parts[3:]) if len(parts) >= 4 else ""
                
                match_ind = any(p.search(series_desc) for p in patterns) if patterns else False
                match_grp = bool(group_pat.search(item_name)) if group_pat else False
                should_convert = match_ind or match_grp

            if should_convert:
                tasks.append(ConversionTask(
                    input_dir=full_path, 
                    output_path=nifti_out_dir / f"{item_name}.nii.gz"
                ))
        return tasks

    @staticmethod
    def process_item(task: ConversionTask) -> Tuple[str, bool, str]:
        """Static method for better cross-platform process serialization."""
        try:
            print(f"Attempting conversion: {task.input_dir} -> {task.output_path}")
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Your original conversion + RAS reorientation logic here
            dicom2nifti.dicom_series_to_nifti(str(task.input_dir), str(task.output_path), reorient_nifti=True)
            
            nii = nib.load(str(task.output_path))
            orig_orient = io_orientation(nii.affine)
            target_orient = axcodes2ornt(('R', 'A', 'S'))
            
            if not np.array_equal(orig_orient, target_orient):
                transform = ornt_transform(orig_orient, target_orient)
                nii_ras = nii.as_reoriented(transform)
                nib.save(nii_ras, str(task.output_path))
                
            return (str(task.input_dir), True, str(task.output_path))
        except Exception as e:
            error_msg = f"Failed conversion for {task.input_dir.name}: {e}"
            print(error_msg)
            if task.output_path.exists():
                try:
                    os.remove(task.output_path)
                except Exception:
                    pass
            return (str(task.input_dir), False, str(e))

    def _report(self, results, start_time):
        success = sum(1 for r in results if r[1])
        print(f"{'-'*20}\nConversion Summary:")
        print(f"  Successful: {success}\n  Failed: {len(results) - success}")
        print(f"  Total Time: {time.time() - start_time:.2f}s\n{'-'*20}")
    
    
    

    


def DICOM_splitter(path : str | Path , max_workers : int = 32, use_only_name : bool = False) -> Tuple[Path, Path]:
    """Splits a potentially directory of Dicom files into nicely named directory one for each scan/document . """
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
        "StudyDescription",
        "SeriesDescription",
        "SeriesNumber"]

    files_to_link = []
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
            stdesc = getattr(dcm, "StudyDescription", "unknownStudy")

            pid = clean_string(str(pid))
            pname = clean_string(str(pname))
            sdesc = clean_string(str(sdesc))
            snum = str(snum)
            stdesc = clean_string(str(stdesc))
            if use_only_name:
                folder = sort_dir / f"{pname}_{pname}_Series{snum}@{stdesc}_{sdesc}"
            else:
                folder = sort_dir / f"{pid}_{pname}_Series{snum}@{stdesc}_{sdesc}"  #some ids are super annoying, so if the have timecodes etc or are super long, and not needed for identifying, use the double name and clean later 

            # Create folder once
            if folder not in known_dirs:
                folder.mkdir(exist_ok=True, parents=True)
                known_dirs.add(folder)

            target = folder / original_file_path.name

            if not target.exists():
                files_to_link.append((original_file_path, target))
            else:
                skipped_files += 1

        except Exception as e:
            print(f"Failed reading {original_file_path}: {e}")
            error_files += 1
    
    # ------------------------------------------------------
    # PASS 2: Link (I/O bound → threads are ideal)
    # ------------------------------------------------------
    print(f"\nStarting parallel link of {len(files_to_link)} files...")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(safe_link, pair) for pair in files_to_link]

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
    print(f"  Linked: {copied_files}")
    print(f"  Skipped: {skipped_files}")
    print(f"  Errors: {error_files}")  #super unlikely unless storage full, never encountered any in testing so far 

    return sort_dir, nifti_out_dir


# --- Helper function for converting a single series ---
def convert_single_series_to_nifti(input_dir_path : str | Path, output_nifti_path : str | Path ) -> Tuple[str, bool, str]:
    """Converts a single directory of DICOM series to a NIFTI file."""
    input_dir_path = Path(input_dir_path)
    output_nifti_path = Path(output_nifti_path)

    try:
        print(f"Attempting conversion: {input_dir_path} -> {output_nifti_path}")
        
        output_nifti_path.parent.mkdir(exist_ok=True)
        dicom2nifti.dicom_series_to_nifti(str(input_dir_path), str(output_nifti_path), reorient_nifti=True )


        # there seems to be no real way to get past the double load if we do not want to rewrite the dicomtonifti source code, so live with the short delay this causes 
        nii = nib.load(str(output_nifti_path)) #type: ignore 
        orig_orient = io_orientation(nii.affine) #type: ignore 
        target_orient = axcodes2ornt(('R', 'A', 'S'))
        if not np.array_equal(orig_orient, target_orient):
            transform = ornt_transform(orig_orient, target_orient)
            nii_ras = nii.as_reoriented(transform) #type: ignore 
            nib.save(nii_ras, str(output_nifti_path)) #type: ignore 
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
# This uses the partial function above, we use ProcessPool as we need a core per conversion, it is quite compute intense and we have no real io wait time, 12 workers seems to be fine, to even handle the whole body scans, keep 4 real cores, for other processes running in the background  
def raw_data_to_nifti_parallel(config : NiftiConfig):
    """
    Sorts DICOMs and converts selected series to NIFTI in parallel.

    Args:
        raw_path (str): Path to the directory containing raw DICOM files.
        scans_indicators (list, optional): List of strings. Only series whose
                                           directory name contains one of these
                                           indicators will be converted (unless use_default=True).
                                           Defaults to None.
        group_filter (str, optional): Just one string for a big group, like a conv kernel for the image (bone/ tissue etc.), will just do a basic substring with that.
        use_default (bool): If True, attempts to convert all series,
                                      ignoring scans_indicators. Defaults to False.
        max_workers (int): Maximum number of processes to use for conversion.
                                     Defaults to 12.
        use_only_name (bool): If True, uses only PatientName for folder naming during DICOM splitting.
                                        Defaults to True.
        max_workers_dicom (int) :  Maximum number of processes to use for DICOM sorting and copying  
    """
    start_time = time.time()
    p = Path(config.raw_path)
    if not p.is_dir():
        print(f"Error: Raw path '{config.raw_path}' not found or is not a directory.")
        return

    # 1. Sort DICOMs sequentially first
    print("-" * 20)
    print("Step 1: Sorting DICOM files...")
    try:
        sort_dir, nifti_out_dir = DICOM_splitter(config.raw_path, max_workers=config.max_workers_dicom, use_only_name=config.use_only_name)
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
    if config.scans_indicators and not config.use_default:
        # Compile regex patterns for faster matching if indicators are provided
        try:
            patterns = [build_pattern(clean_string(indicator)) for indicator in config.scans_indicators]
            print(f"Using scan indicators: {config.scans_indicators}")
        except re.error as e:
            print(f"Error compiling regex for scan indicators: {e}. Please check indicators: {config.scans_indicators}")
            return 
    if config.group_filter and not config.use_default:
        try:
            group_pattern = re.compile(re.escape(config.group_filter), re.IGNORECASE)
            print(f"Using group filter: '{config.group_filter}'")
        except re.error as e:
            print(f"Error compiling regex for group filter: {e}. Please check filter: {config.group_filter}")
            return

    skipped_dirs = []
    dirs_to_convert = []

    for item_name  in os.listdir(sort_dir):  #keep the listdir here, we need the individual name, using iterdir would neeed .name everywhere 
        full_path = sort_dir / item_name
        if full_path.is_dir():
            parts = item_name.split("_")
            should_convert = False
            if config.use_default:
                should_convert = True
            else: 
                match_indicator = False
                match_group = False

                if patterns:
                    if len(parts) >= 4:
                        series_description_from_name = "_".join(parts[3:])  # The description is everything from the 4th element onwards, joined by underscores, it was the best solution to mget usuable file names for downstream uses
                        if any(p.search(series_description_from_name) for p in patterns):
                            match_indicator = True
                    elif len(parts) < 4:
                        print(f"Unexpected name format {item_name} for indicator check")
                
                
                if group_pattern:
                    if group_pattern.search(item_name):  #type: ignore 
                        match_group = True
                
                # Convert if *either* filter matches (and filters are active) we do not want both active at the same time 
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
    print(f"Step 3: Starting parallel conversion using up to {config.max_workers} processes...")
    results = []
    successful_conversions = 0
    failed_conversions = 0

    # Use ProcessPoolExecutor
    # The 'with' statement ensures the pool is properly shut down
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all tasks. submit returns a Future object.
        future_to_input = {executor.submit(convert_single_series_to_nifti, task[0], task[1]): task[0] for task in tasks}  #this allows for good error handling and matching what files did not work, keep the dict and not use a list here  

        
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





def modify_metadata(dcm_file : str | Path, backup : bool =True, custom : bool | None  = False, new_desc : str | None = None , new_name : str | None = None ):
    """Modifies metadata of a single file"""
    # For more Keywords, tags see https://dicom.innolitics.com/ciods/rt-dose/patient/00100010
    try:
        ds = pydicom.dcmread(dcm_file)
        
        
        if backup:
            ds.save_as(dcm_file.replace('.dcm', '_backup.dcm')) #type: ignore 
        
        
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

def modify_dcms(directory_path : str | Path, new_desc : str | None , new_name : str | None , custom : bool | None  ):
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
def nifti_renamer(nii_path : str | Path , prefix : str | None , suffix: str | None , number : int , file_mapping : dict):
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
    config = NiftiConfig(raw_path=Path(r"C:\Users\schum\Desktop\zesbo\test_ank\anke"), max_workers=6, max_workers_dicom=12, use_default=True)
    converter = NiftiParallelConverter(config)
    converter.run()