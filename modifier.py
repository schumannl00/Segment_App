import os
import shutil 
from pathlib import Path 
import logging

def merger(root_dir):
    target_dir= Path(root_dir.parent)/ 'merged'
    os.makedirs(target_dir, exist_ok=True)
    processed_files = 0 
    for root, dirs, files in os.walk((os.path.normpath(root_dir)), topdown=False):
        rel_path = os.path.relpath(root, root_dir)
        for name in files: # removed dcm failsafe as Data from e.g. Sectra is missing the .dcm file extensions for some reason
            processed_files += 1
            if rel_path != "":
                path_prefix=rel_path.replace(os.sep, "_")
                new_name = F"{path_prefix}_{name}"
            else:
                new_name = name 
            source_file = os.path.join(root, name)
            target_file = os.path.join(target_dir, new_name)
            shutil.copy2(source_file, target_file)
    print(processed_files)
    return target_dir


def stl_renamer_with_lut(stl_output_path : Path , file_mapping : dict ):
    lookup = {coded["number"]: original_filename for original_filename, coded in file_mapping.items()}
    renamed_dirs=[]
    for item in os.listdir(stl_output_path):
        item_path = os.path.join(stl_output_path, item)
        
      
        if not os.path.isdir(item_path):
            continue
        
        
        if item.startswith('STL') and len(item) >= 6:
            # Extract the number part (3 digits after 'STL')
            number = item[3:6]
            
            # Rest of the directory name (if any)
            rest_of_name = item[6:] if len(item) > 6 else ""
                
            # Check if number exists in lookup
            if number in lookup:
                original_name = lookup[number].replace(".nii.gz", "")
                # Create new directory name (preserving any suffix)
                new_dir_name = f"{original_name}{rest_of_name}"
                new_dir_path = os.path.join(stl_output_path, new_dir_name)
                try: 
                    for file in os.listdir(item_path):
                        old_file_path = os.path.join(item_path, file)
                        if os.path.isfile(old_file_path):
                            new_file_name = original_name + "_" + file
                            new_file_path = os.path.join(item_path, new_file_name)
                            shutil.move(old_file_path, new_file_path)

                except Exception as e:
                    print(f"{str(e)}")
                try:
                    # Check if destination already exists
                    if os.path.exists(new_dir_path):
                        logging.warning(f"Destination {new_dir_path} already exists. Skipping.")
                        continue
                    
                    # Perform the rename operation using shutil.move
                    logging.info(f"Renaming: {item_path} -> {new_dir_path}")
                    shutil.move(item_path, new_dir_path)
                    renamed_dirs.append((item, new_dir_name))
                    
                except Exception as e:
                    logging.error(f"Error renaming {item_path}: {str(e)}")
            else:
                logging.warning(f"No matching original filename found for number {number}")
