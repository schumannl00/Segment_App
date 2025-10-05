import os 
import numpy as np
import nibabel as nib
from nibabel import load, Nifti1Image, save 
import shutil

import pathlib
from pathlib import Path


#img1=nib.load(r"D:\nnUNet_raw\Dataset111_gesund\labelsTr\Syn_015.nii.gz")
#img2=nib.load(r"D:\nnUNet_raw\Dataset111_gesund\imagesTr\Syn_015_0000.nii.gz")
#print(img1.shape)
#print(img2.shape)

#print(np.unique(img.get_fdata()))

def cut(nii_path1, nii_path2, lower, upper, dump_folder):
    nii1= load(nii_path1)
    nii2= load(nii_path2)
    suffix = "cut"
    x_range=slice(0,nii2.shape[0])
    y_range =slice(0,nii2.shape[1])
    z_range= slice(0,nii2.shape[2])
    cut_z_range = slice(lower, upper)
    if z_range == slice(0,nii1.shape[2]):
        print("Dimensions should match")
    roi1 = np.asarray(nii1.dataobj[x_range,y_range,cut_z_range])
    roi2 = np.asarray(nii2.dataobj[x_range,y_range,cut_z_range])
    img1= Nifti1Image(roi1,affine=nii1.affine)
    img2 = Nifti1Image(roi2,affine=nii2.affine)
    basename1, ext1 = nii1.get_filename().split(os.extsep, 1)
    basename2, ext2 = nii2.get_filename().split(os.extsep, 1)
    out_name1= f'{basename1}_{suffix}.{ext1}'
    out_name2= f'{basename2}_{suffix}.{ext2}'
    save(img1, out_name1)
    save(img2, out_name2)
  
def zcut(nii_path : str, lower : int, upper : int , keep_original: bool, backup_dir : str ) -> None :
    nii = load(nii_path)
    suffix = "cut"
    x_range=slice(0,nii.shape[0])
    y_range =slice(0,nii.shape[1])
    max_z = nii.shape[2]
    upper = min(upper, max_z)
    cut_z_range = slice(lower, upper)
    roi = np.asarray(nii.dataobj[x_range,y_range,cut_z_range])
    affine = nii.affine.copy()
    if lower > 0:
        offset = np.array([0, 0, lower, 1])
        new_origin = nii.affine @ offset
        affine[:3, 3] = new_origin[:3]
    img = Nifti1Image(roi, affine=affine)
    basename, ext = nii.get_filename().split(os.extsep, 1)
    out_name= f'{basename}_{suffix}.{ext}'
    save(img, out_name)
    if keep_original:
        backup_dir= backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        shutil.move(nii_path, backup_dir)
    else: os.remove(nii_path)

def masking(nii_path):
    nib.openers.Opener.default_compresslevel = 9
    nii = load(nii_path)
    volume =  nii.get_fdata()
    affine = nii.affine
    x_split= volume.shape[0]//2
    left_data=volume.copy()
    left_data[:x_split,:,:]=0 #zeros out left side of the images which corresponds to the patients right side in the usual position
    suffix_right = 'rechts'
    suffix_left= 'links'
    right_data= volume.copy()
    right_data[x_split:,:,:] =0
    img_rechts = Nifti1Image(right_data.astype(np.int16), affine)
    img_links = Nifti1Image(left_data.astype(np.int16), affine)
    basename, ext = nii.get_filename().split(os.extsep, 1)
    out_name_rechts = f'{basename}-{suffix_right}.{ext}'
    out_name_links = f'{basename}-{suffix_left}.{ext}'
    save(img_rechts, out_name_rechts)
    save(img_links, out_name_links)
    os.remove(nii_path)



