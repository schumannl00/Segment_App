import json




def rename_keys(original_json, output_json, mapping_dict):
    try:
       
        with open(original_json, 'r') as f:
            data = json.load(f)
        
        new_data = {}

        for old_key, value in data.items():
            # Split key: 'XXX_number_bonepart' -> ['XXX', 'number', 'bonepart']
            parts = old_key.split('_')
            
            if len(parts) >= 2:
                prefix = parts[0]   # XXX
                number = parts[1]   # number
                suffix = "_".join(parts[2:]) if len(parts) > 2 else ""
                new_name = mapping_dict.get(number, number).replace('.nii.gz', '')
                cleaned_new_name = new_name.split('.')[0]
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
    original_json_path = r"E:\test_stemlit\stl_metadata.json"
    output_json_path = r"E:\test_stemlit\stl_metadata_renamed.json"

    
    mapping_dict = {'001': '57-25L-SynG3_57-25L-SynG3_Series16_OSG-61-0,80-Br64-A3-KF.nii.gz', '002': '57-25L-SynG3_57-25L-SynG3_Series21_OSG-63-0,80-Br40-A1-WT.nii.gz', '003': '57-25L-SynG3_57-25L-SynG3_Series23_OSG-64-0,80-Br40-A1-WT.nii.gz', '004': '57-25L-SynG3_57-25L-SynG3_Series28_OSG-67-0,80-Br64-A3-KF.nii.gz', '005': '57-25L-SynG3_57-25L-SynG3_Series35_OSG-70-0,80-Br40-A1-WT.nii.gz', '006': '57-25L-SynG3_57-25L-SynG3_Series41_OSG-R1-III-0,80-Br40-A1-WT.nii.gz'}
    rename_keys(original_json_path, output_json_path, mapping_dict)