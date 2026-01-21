import json
from typing import List, Union 
from pathlib import Path 
PathLike = Union[str, Path]

def rename_keys(original_json : PathLike, output_json: PathLike, mapping_dict : dict):
    try:
       
        with open(original_json, 'r') as f:
            data = json.load(f)
        
        new_data = {}

        for old_key, value in data.items():
            # Split key: 'XXX_number_bonepart' -> ['XXX', 'number', 'bonepart']
            parts = old_key.split('_')
            
            if len(parts) >= 2:
                prefix = parts[0]   # XXX
                number = parts[1] 
                cleaned_number , side  = number.split("-") if "-" in number else (number, "")
                print(cleaned_number)
                final_side = "_" + side.upper() 
                suffix = "_".join(parts[2:]) if len(parts) > 2 else ""
                new_name = mapping_dict.get(cleaned_number, cleaned_number).replace('.nii.gz', '')
                print(f"New name: {new_name}")
                cleaned_new_name = new_name.split('.')[0] + final_side
                new_key = f"{cleaned_new_name}_{suffix}" if suffix else new_name
                new_data[new_key] = value
            else:
                new_data[old_key] = value

      
        with open(output_json, 'w') as f:
            json.dump(new_data, f, indent=4)
            
        print(f"Successfully renamed {len(new_data)} keys and saved to {output_json}")

    except Exception as e:
        print(f"Error: {e}")

    return output_json

if __name__ == "__main__":
    original_json_path = r"E:\fix_orientation\raw\stl_metadata.json"
    output_json_path = r"E:\fix_orientation\raw\stl_metadata_fixed.json"

    
    mapping_dict = {'001': "Anonymous-Female-1975_Anonymous-Female-1975_Series201_1mm-x,-iDose-(3).nii.gz", } 
    rename_keys(original_json_path, output_json_path, mapping_dict)