import os 
import numpy as np
import nibabel as nib
from nibabel import load, Nifti1Image, save 
import shutil

import pathlib
from pathlib import Path

def cut_volume(
    nii_path: str,
    lower: tuple[int | None, int | None, int | None],
    upper: tuple[int | None, int | None, int | None], 
    keep_original: bool,
    destination_dir: str,
    localiser : str = "cut" , 
) -> str:
    """
    Cut a NIfTI volume along x, y, z axes.

    Parameters
    ----------
    nii_path : str
        Path to NIfTI file.
    lower : tuple
        Lower bounds (x, y, z). Use None or 0 to leave axis uncut from below.
    upper : tuple
        Upper bounds (x, y, z). Use None to leave axis uncut from above.
    keep_original : bool
        If True, move original to backup_dir. If False, delete original.
    backup_dir : str
        Directory for backup if keep_original is True.
    """
    os.makedirs(destination_dir, exist_ok = True)
    nii = load(nii_path)
    shape = nii.shape[:3]


    slicers = []
    for ax in range(3):
        lo = 0 if (lower[ax] is None) else lower[ax]
        hi = shape[ax] if (upper[ax] is None) else min(upper[ax], shape[ax])
        slicers.append(slice(lo, hi))

    roi = np.asarray(nii.dataobj[tuple(slicers)])
    affine = nii.affine.copy()


    offset = np.array([
        0 if lower[0] is None else lower[0],
        0 if lower[1] is None else lower[1],
        0 if lower[2] is None else lower[2],
        1
    ], dtype=float)
    if np.any(offset[:3] > 0):
        new_origin = nii.affine @ offset
        affine[:3, 3] = new_origin[:3]

    p = Path(nii_path)
    out_path = Path(destination_dir)/ f"{p.stem}_{localiser}{''.join(p.suffixes)}"
    img = Nifti1Image(roi, affine=affine)
    save(img, str(out_path))

    if not keep_original:
        os.remove(nii_path)
      
        
    return str(out_path)

def sep(nii_path : str, x_cut : int,  destination_dir : str ): 
    # Right half (array left → anatomy right)
    right_file = cut_volume(
        nii_path=nii_path,
        lower=(None, None, None),
        upper=(x_cut, None, None),
        keep_original=True,
        destination_dir= destination_dir,
        localiser="right"
    )
    print("right done")
    # Left half (array right → anatomy left)
    left_file = cut_volume(
        nii_path=nii_path,
        lower=(x_cut, None, None),
        upper=(None, None, None),
        keep_original=True,
        destination_dir= destination_dir,
        localiser="left"
    )
    print("finished")

    return left_file, right_file

#might just be replaced with cut where x,y None for all
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


#this is not the best option but worked so far, cut also properly cut the images, some viewers might struggle with the overlay, and if stuff ges wrong, simply adding is easier
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



if __name__ == "__main__":
    cut_volume(r"\\wsl.localhost\Ubuntu\home\schumannl\datasets\rib_prep\nii\Sco_056_0000.nii.gz", lower=(60, None, 370), upper=(470, None, 890), keep_original=True, destination_dir=r"\\wsl.localhost\Ubuntu\home\schumannl\datasets\rib_prep\nii\cut_test")