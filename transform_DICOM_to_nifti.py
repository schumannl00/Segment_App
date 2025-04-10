import os
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import dicom2nifti
import pydicom 
import dicom2nifti.settings as settings 
import glob 
import shutil
import re
import json

#Trennt gesammelte dicom daten in die entsprechenden scans, sortiert die problematischen aus und convertiert diese dann in den NIFTI ordner. 

settings.disable_validate_slice_increment()
settings.enable_resampling()
settings.set_resample_spline_interpolation_order(1)
settings.set_resample_padding(-1000)

#dicom2nifti.convert_directory(path_file, out_path)

#for filename in glob.iglob(f'{path}/*'):
    #dicom2nifti.dicom_series_to_nifti(filename, os.path.join(out_path,f"{filename}" +".nii.gz"))

#dicom2nifti.dicom_series_to_nifti(path, os.path.join(out_path, "MPR COR LI OSG.nii.gz"))

#modify with filter argument PatientID

def DICOM_splitter(path):
      p= Path(path)
      os.makedirs(os.path.join(str(p.parent),'sortiert'), exist_ok=True)
      os.makedirs(os.path.join(str(p.parent), 'NIFTI'), exist_ok=True)
      for f in os.listdir(path):
         original_file_path = os.path.join(path,f)
         if os.path.isfile(os.path.join(path,f)):
                file_name = os.path.basename(original_file_path)
                read_file= pydicom.dcmread(original_file_path)
                if hasattr(read_file, 'SeriesDescription') and hasattr(read_file,"Modality"):
                    file_series_description = read_file.SeriesDescription
                    file_modality = read_file.Modality
                    if not read_file.PatientID:
                        file_series_id = read_file.PatientName
                    else: file_series_id = read_file.PatientID
                    description_path = os.path.join(os.path.join(str(p.parent),'sortiert'), str(file_series_id) + '_'+ file_modality + "_" + file_series_description)
                else:
                    file_series_id = read_file.PatientID
                    file_modality = read_file.Modality
                    description_path = os.path.join(os.path.join(str(p.parent),'sortiert'), file_series_id + file_modality)
                if not os.path.exists(description_path):
                    os.makedirs(description_path)
                if hasattr(read_file, 'SeriesDescription'):  
                    if read_file.SeriesDescription.strip():
                        shutil.copyfile(original_file_path, os.path.join(description_path, file_name))
                else: shutil.copyfile(original_file_path, os.path.join(description_path, file_name))



def raw_data_to_nifti(raw_path, scans_indicators=None, use_default = False):
    p=Path(raw_path)
    DICOM_splitter(raw_path)
    path = os.path.join(str(p.parent),'sortiert')
    #out_path = os.path.join(str(p.parent),r'NIFTI')
    if scans_indicators != None:
        patterns = [re.compile(r"\b" + re.escape(indicator) + r"\b") for indicator in scans_indicators]
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if Path(full_path).is_dir():
            if (not use_default) and any(pattern.search(filename) for pattern in patterns):
                print(filename)
                nifti_path =  os.path.join(str(p.parent),r'NIFTI',f"{filename}" + ".nii.gz")
                dicom2nifti.dicom_series_to_nifti(full_path, nifti_path)
            elif (not use_default) and any(s in filename for s in scans_indicators)==False:
                print(f"{filename} is not converted as it does not match any of the Indicators")
            else: 
                try:
                    nifti_path =  os.path.join(str(p.parent),r'NIFTI',f"{filename}" + ".nii.gz")
                    dicom2nifti.dicom_series_to_nifti(full_path, nifti_path)
                except: print("There are files that cannot be converted. Please select appropriate scan indicators. Usually the Dosis Info or an Exam Summary causes problem. Check the sortiert folder for the problematic folders or files.")
                 

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
    

#if __name__ == "__main__":
    #raw_data_to_nifti(r"D:\nnUNet_raw\Dataset116_Becken\sortiert", use_default=True)
    #interference_path = ""
    #file_mapping = {}
    #for i, folder in enumerate(os.listdir(interference_path)):
        #nifti_renamer(os.path.join(interference_path,folder), prefix='Bck_', suffix='', number=i, file_mapping=file_mapping)
    #json_path = os.path.join(Path(interference_path).parent, 'decoder_label.json')
    #with open(json_path, 'w') as json_file:
        #json.dump(file_mapping, json_file, indent=4)
