import os 
import numpy as np
import nibabel as nib
from nibabel import load, Nifti1Image, save #type: ignore 
import shutil
import pathlib
from pathlib import Path
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from typing import List, Tuple, Union 
PathLike = Union[str, Path]

def cut_volume(
    nii_path: PathLike ,
    lower: Tuple[int | None, int | None, int | None],
    upper: Tuple[int | None, int | None, int | None], 
    keep_original: bool,
    destination_dir: PathLike,
    localiser : str = "cut" , percents_given: bool = False, input_type : str = "RAS") -> str:
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
    nii = nib.load(nii_path) #type: ignore 
    
    # 1. Standardize Orientation
    # Get current orientation of the file
    orig_ornt = io_orientation(nii.affine) #type: ignore 
    
    # Define our target: RAS+ (Standard Neurological)
    target_ornt = axcodes2ornt(('R', 'A', 'S'))
    
    # If the file isn't already RAS, reorient it
    if not np.array_equal(orig_ornt, target_ornt):
        print(f"Reorienting from {orig_ornt} to {target_ornt}")
        transform = ornt_transform(orig_ornt, target_ornt)
        nii = nii.as_reoriented(transform) #type:ignore
    
    # Now that we are in RAS: 
    # Index 0 is Left->Right, 1 is Post->Ant, 2 is Inf->Sup
    shape = nii.shape #type: ignore 
    processed_lower = []
    processed_upper = []
    safe_lower = lower if lower is not None else [None, None, None]
    safe_upper = upper if upper is not None else [None, None, None]
    for i in range(3):
        dim_max = shape[i]
        val_low = safe_lower[i]
        val_high = safe_upper[i]
        # 1. Handle None defaults first
        l_input = val_low if val_low is not None else 0
        u_input = val_high if val_high is not None else (100 if percents_given else dim_max)
        
        # 2. Convert to absolute RAS pixels

        l_px: int # Declare the type explicitly
        u_px: int
        if percents_given:
            l_px = int(dim_max * (l_input / 100.0))
            u_px = int(dim_max * (u_input / 100.0))
        else:
            l_px = int(l_input)
            u_px = int(u_input)

        # 3. Handle LPS Flip (X and Y only)
        # In LPS, the axis is mirrored. Start becomes (Max - End)
        if input_type == "LPS" and i < 2:
            final_l = dim_max - u_px
            final_u = dim_max - l_px
        else:
            final_l = l_px
            final_u = u_px

        # 4. Final safety clip to image boundaries
        processed_lower.append(max(0, min(final_l, dim_max)))
        processed_upper.append(max(0, min(final_u, dim_max)))

    # Construct slices for dataobj
    final_coords = [slice(l, u) for l, u in zip(processed_lower, processed_upper)]
    # 3. Efficient slicing using dataobj (Lazy Load)
    slicers = tuple(final_coords)
    roi = np.asarray(nii.dataobj[slicers]) #type: ignore
    affine = nii.affine.copy() #type: ignore 

    offset = np.array([0 if s.start is None else s.start for s in final_coords ] + [1])
    print("Offset:", offset)
    print("Slicers:", slicers)
    if np.any(offset[:3] > 0):
        new_origin = nii.affine @ offset #type: ignore 
        affine[:3, 3] = new_origin[:3]

    p = Path(nii_path)
    out_path = Path(destination_dir)/ f"{p.stem}_{localiser}{''.join(p.suffixes)}"
    img = Nifti1Image(roi, affine=affine)
    save(img, str(out_path))

    if not keep_original:
        os.remove(nii_path)
      
        
    return str(out_path)




def sep(nii_path : PathLike, x_cut : int,  destination_dir : PathLike ): 
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

#depreceated function, use cut_volume instead
def zcut(nii_path : str, lower : int, upper : int , keep_original: bool, backup_dir : str ) -> None :
    nii = load(nii_path)
    suffix = "cut"
    x_range=slice(0,nii.shape[0]) #type: ignore 
    y_range =slice(0,nii.shape[1]) #type: ignore 
    max_z = nii.shape[2] #type: ignore 
    upper = min(upper, max_z)
    cut_z_range = slice(lower, upper)
    roi = np.asarray(nii.dataobj[x_range,y_range,cut_z_range]) #type: ignore 
    affine = nii.affine.copy() #type: ignore 
    if lower > 0:
        offset = np.array([0, 0, lower, 1])
        new_origin = nii.affine @ offset #type: ignore 
        affine[:3, 3] = new_origin[:3]
    img = Nifti1Image(roi, affine=affine)
    basename, ext = nii.get_filename().split(os.extsep, 1) #type: ignore 
    out_name= f'{basename}_{suffix}.{ext}'
    save(img, out_name)
    if keep_original:
        backup_dir= backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        shutil.move(nii_path, backup_dir)
    else: os.remove(nii_path)


#this is not the best option but worked so far, cut also properly cut the images, some viewers might struggle with the overlay, and if stuff ges wrong, simply adding is easier
def masking(nii_path):
    nib.openers.Opener.default_compresslevel = 9 #type: ignore 
    nii = load(nii_path)
    volume =  nii.get_fdata() #type: ignore 
    affine = nii.affine #type: ignore 
    x_split= volume.shape[0]//2
    left_data=volume.copy()
    left_data[x_split:,:,:]=0 #zeros out left side of the images which corresponds to the patients right side in the usual position, not with ras anymore 
    suffix_right = 'rechts'
    suffix_left= 'links'
    right_data= volume.copy()
    right_data[:x_split,:,:] =0
    img_rechts = Nifti1Image(right_data.astype(np.int16), affine)
    img_links = Nifti1Image(left_data.astype(np.int16), affine)
    basename, ext = nii.get_filename().split(os.extsep, 1) #type: ignore 
    out_name_rechts = f'{basename}-{suffix_right}.{ext}'
    out_name_links = f'{basename}-{suffix_left}.{ext}'
    save(img_rechts, out_name_rechts)
    save(img_links, out_name_links)
    os.remove(nii_path)



if __name__ == "__main__":
    cut_volume(r"E:\becken2710\NIFTI\Bck_001_0000.nii.gz", lower = (30, 20, 20), upper = (70, 70, 90), keep_original= True, destination_dir= r"E:\becken2710\NIFTI\cut_test", percents_given= True,) 