import os
import shutil 
from pathlib import Path 

def merger(root_dir):
    target_dir= Path(root_dir.parent)/ 'merged'
    os.makedirs(target_dir, exist_ok=True)
    processed_files = 0 
    for root, dirs, files in os.walk((os.path.normpath(root_dir)), topdown=False):
        rel_path = os.path.relpath(root, root_dir)
        for name in files: # removed dcm failsafe
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