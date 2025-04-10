import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import re
import sys
import json
import pathlib
from pathlib import Path
import numpy as np
import dicom2nifti
import pydicom 
import dicom2nifti.settings as settings 
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nibabel as nib
from nibabel import load, Nifti1Image, save 
from skimage import measure
from scipy import ndimage
import pyvista as pv
from stl import mesh
from nii_to_stl_final import convert_to_LPS, process_with_parameters, process_directory
from transform_DICOM_to_nifti import DICOM_splitter, raw_data_to_nifti, nifti_renamer
import shutil
from leg_seperator import separator, zcut

id_dict = {
    '111': {
        'body_part': 'Sprunggelenk_gesund',
        'Path_to_results': {'3d_fullres' : "Dataset111_gesund/nnUNetTrainer__nnUNetPlans__3d_fullres"},
        'configurations': ['3d_fullres'],
        'prefix' : "Syn_",
        'suffix': '_0000'
    },
    '112': {
        'body_part': 'Sprunggelenk_mixed',
        'Path_to_results': {'3d_fullres' : "Dataset112_mixed/nnUNetTrainer__nnUNetPlans__3d_fullres"},
        'configurations': ['3d_fullres'],
        'prefix' : "Syn_",
        'suffix': '_0000'
    },
    '113': {
        'body_part': 'Sprunggelenk_beideFüße_DONT_USE',
        'Path_to_results': {'3d_fullres' : 'TBA' },
        'configurations': ['3d_fullres'],
        'prefix' : "Syn_",
        'suffix': '_0000'
    },
    '114':{
      'body_part': 'Schulter',
        'Path_to_results': {'3d_fullres' : "Dataset114_Schulter/nnUNetTrainer__nnUNetPlans__3d_fullres"},
        'configurations': ['3d_fullres'],
        'prefix' : "Sch_",
        'suffix': '_0000'
    },
    '116': {
        'body_part': 'Becken',
        'Path_to_results': {'3d_fullres' : "Dataset116_Becken/nnUNetTrainer__nnUNetPlans__3d_fullres" },
        'configurations': ['3d_fullres'],
        'prefix' : "Bck_",
        'suffix': '_0000'
    }
    }

labels_dict = {
    '111': {
        1: "Fibula",
        2: "Talus",
        3: "Tibia"
    },
    '112': {
         1: "Fibula",
        2: "Talus",
        3: "Tibia"
    },
    '113': {
        1: 'Fibula links',
        2: 'Fibula rechts',
        3: 'Talus links',
        4: 'Talus rechts',
        5: 'Tibia links',
        6: 'Tibia rechts'
     },
     '114': {1: 'Schulter'},
     '116': {1: 'Becken'}
}

# Default segment parameters -  smoothing for labelmap just removes salt-pepper noise, taubin maintains volume with  light smoothing-params to remove some steps\artifacts 
default_segment_params = {
    'smoothing': 0.05, 
    'mesh_smoothing_method': 'taubin', 
    'mesh_smoothing_iterations': 35, 
    'mesh_smoothing_factor': 0.05
}

# Initialize segment_params with defaults for all labels in labels_dict
segment_params = {}
for id_key, labels in labels_dict.items():
    if id_key not in segment_params:
        segment_params[id_key] = {}
    for label_id, label_name in labels.items():
        segment_params[id_key][label_id] = {
            'label': label_name,
            **default_segment_params
        }

class ParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parameter Input GUI for nnUNet segmentation")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        
        # Create main frame with padding
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Row 1: Input Path with drag and drop
        ttk.Label(main_frame, text="Input Path:").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        self.input_path = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path, width=40)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.drop_target_register(DND_FILES)
        self.input_entry.dnd_bind('<<Drop>>', lambda e: self.drop_input_file(e, self.input_entry)) 
        input_button = ttk.Button(input_frame, text="Browse", command=self.browse_input_path)
        input_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Row 2: Output Paths (STL and Labelmaps)
        ttk.Label(main_frame, text="STL Output Path:").grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        stl_output_frame = ttk.Frame(main_frame)
        stl_output_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        self.stl_output_path = tk.StringVar()
        self.stl_output_entry = ttk.Entry(stl_output_frame, textvariable=self.stl_output_path, width=40)
        self.stl_output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stl_output_entry.drop_target_register(DND_FILES)
        self.stl_output_entry.dnd_bind('<<Drop>>', lambda e: self.drop_output_files(e, self.stl_output_entry))
        stl_output_button = ttk.Button(stl_output_frame, text="Browse", command=self.browse_stl_output)
        stl_output_button.pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Label(main_frame, text="nnUNet Labelmap Output Path:").grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        labelmap_output_frame = ttk.Frame(main_frame)
        labelmap_output_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        self.labelmap_output_path = tk.StringVar()
        self.labelmap_output_entry = ttk.Entry(labelmap_output_frame, textvariable=self.labelmap_output_path, width=40)
        self.labelmap_output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.labelmap_output_entry.drop_target_register(DND_FILES)
        self.labelmap_output_entry.dnd_bind('<<Drop>>',  lambda e: self.drop_output_files(e, self.labelmap_output_entry))
        labelmap_output_button = ttk.Button(labelmap_output_frame, text="Browse", command=self.browse_labelmap_output)
        labelmap_output_button.pack(side=tk.RIGHT, padx=(5, 0))

        # Row 3: Scan Indicators (as set of strings)
        indicator_frame = ttk.Frame(main_frame)
        indicator_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(indicator_frame, text="Scan Indicators:").pack(side=tk.LEFT, padx=(0, 5))

        # Create a StringVar to store the selected indicators
        self.scan_indicators = tk.StringVar()

        # Create a set to store the selected indicators
        self.selected_indicators = set()

        # menubutton
        self.indicators_menu = ttk.Menubutton(indicator_frame, text="Select Indicators", width=40)
        self.indicators_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

        #dropdown menu
        self.dropdown_menu = tk.Menu(self.indicators_menu, tearoff=0)
        self.indicators_menu["menu"] = self.dropdown_menu

        # Custom Indicator Button
        self.add_custom_button = ttk.Button(
            indicator_frame, 
            text="+", 
            width=3, 
            command=self.add_custom_indicator)
        
        self.add_custom_button.pack(side=tk.LEFT, padx=(5, 0))


        # Checkbox for using default indicators
        self.use_default_indicators = tk.BooleanVar(value=False)
        self.default_indicators_check = ttk.Checkbutton(
            indicator_frame, 
            text="Don't filter", 
            variable=self.use_default_indicators,
            command=self.toggle_indicators_entry
        )
        self.default_indicators_check.pack(side=tk.RIGHT, padx=(5, 0))

       
        

        # Row 4: Select ID and Configurations
        ttk.Label(main_frame, text="ID:").grid(row=4, column=0, sticky=tk.W, pady=(0, 10))
        self.id_var = tk.StringVar()
        self.id_var.trace_add('write', self.on_key_change)
        id_combobox = ttk.Combobox(main_frame, textvariable=self.id_var, width=40)
        id_combobox['values'] = list(id_dict.keys())
        id_combobox.grid(row=4, column=1, sticky=tk.W, pady=(0, 10))
        id_combobox.bind("<<ComboboxSelected>>", self.update_configurations)
        
        ttk.Label(main_frame, text="Dataset name:").grid(row=5, column=0, sticky=tk.W, pady=(0, 10))
        self.datasetname = tk.StringVar()
        self.datasetname.trace_add('write', self.on_namechange)
        self.datasetname.set("Select Name.")
        name_combobox= ttk.Combobox(main_frame, textvariable= self.datasetname, width= 40)
        name_combobox['values'] = [id_dict[id_key]['body_part'] for id_key in list(id_dict.keys())]
        name_combobox.grid(row=5, column=1, sticky=tk.W, pady=(0, 10))
        name_combobox.bind("<<ComboboxSelected>>", self.on_namechange)

        

        ttk.Label(main_frame, text="Configuration:").grid(row=6, column=0, sticky=tk.W, pady=(0, 10))
        self.config_var = tk.StringVar()
        self.config_combobox = ttk.Combobox(main_frame, textvariable=self.config_var, width=40)
        self.config_combobox.grid(row=6, column=1, sticky=tk.W, pady=(0, 10))

        # Row 5: Folds (check boxes)
        ttk.Label(main_frame, text="Folds:").grid(row=7, column=0, sticky=tk.W, pady=(0, 10))
        self.folds_frame = ttk.Frame(main_frame)
        self.folds_frame.grid(row=7, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        self.folds_var = tk.StringVar(value="0,1,2,3,4")  # Default value of 5 folds
        self.folds_checkbuttons = []
        for i in range(5):
            var = tk.BooleanVar(value=True)  # Default to selected
            check = ttk.Checkbutton(self.folds_frame, text=f"Fold {i}", variable=var)
            check.grid(row=0, column=i, padx=5)
            self.folds_checkbuttons.append((var, i))

        self.select_all_button = ttk.Button(self.folds_frame, text="Select All", command=self.select_all_folds)
        self.select_all_button.grid(row=1, column=0, columnspan=5)

       #Row 6
        self.split_nifti_var = tk.BooleanVar(value=False)
        self.split_nifti_check = ttk.Checkbutton(main_frame, text="Split NIFTIs into left and right", variable=self.split_nifti_var)
        self.split_nifti_check.grid(row=8, column=1, sticky=tk.W, pady=(0, 10))
        #Row 7
        self.enable_zcut = tk.BooleanVar(value=False)
        enable_checkbox = ttk.Checkbutton(main_frame, 
                                         text="Cut along z direction",
                                         variable=self.enable_zcut,
                                         command=self.toggle_zcut_inputs)
        enable_checkbox.grid(row=9, column=1, columnspan=4, sticky=tk.W, pady=(0, 10))
        
        # Z-range parameters (single row, two columns)
        ttk.Label(main_frame, text="Z-Range:").grid(row=9, column=0, sticky=tk.W, pady=(0, 10))
        
        # Lower bound
        ttk.Label(main_frame, text="Lower:").grid(row=9, column=2, sticky=tk.E, pady=(0, 10), padx=(0, 5))
        self.lower_var = tk.StringVar(value="0")
        self.lower_entry = ttk.Entry(main_frame, textvariable=self.lower_var, width=10)
        self.lower_entry.grid(row=9, column=3, sticky=tk.W, pady=(0, 10))
        self.lower_entry["state"] = "disabled"
        
        # Upper bound
        ttk.Label(main_frame, text="Upper:").grid(row=9, column=4, sticky=tk.E, pady=(0, 10), padx=(20, 5))
        self.upper_var = tk.StringVar(value="0")
        self.upper_entry = ttk.Entry(main_frame, textvariable=self.upper_var, width=10)
        self.upper_entry.grid(row=9, column=5, sticky=tk.W, pady=(0, 10))
        self.upper_entry["state"] = "disabled"

        #Row 8
        self.multiple_ids_var = tk.BooleanVar(value=False)
        self.multiple_ids_check =ttk.Checkbutton(main_frame, text="Does the Input Folder have indiviudual subfolders \n e.g. multiple patients? Use custom filters ('+' -Button) as \n naming conventions might vary.", variable=self.multiple_ids_var)
        self.multiple_ids_check.grid(row=13, column=1, sticky=tk.W, pady=(0, 10))

        # Button frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=14, column=0, columnspan=2, pady=(20, 0))
        submit_button = ttk.Button(button_frame, text="Submit", command=self.submit)
        submit_button.pack(side=tk.LEFT, padx=(0, 10))
        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_fields)
        clear_button.pack(side=tk.LEFT)

        # Progress bar for displaying operation progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=15, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Segment parameters editing window
        self.segment_params_button = ttk.Button(button_frame, text="Edit Segment Params", command=self.edit_segment_params)
        self.segment_params_button.pack(side=tk.LEFT)

        # Configure grid to expand with window resizing
        main_frame.columnconfigure(1, weight=1)

        # Status bar at bottom of window
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

    
    def toggle_zcut_inputs(self):
        """Enable or disable Z-cut inputs based on checkbox state"""
        if self.enable_zcut.get():
            self.lower_entry["state"] = "normal"
            self.upper_entry["state"] = "normal"
        else:
            self.lower_entry["state"] = "disabled"
            self.upper_entry["state"] = "disabled"

    def toggle_indicator(self, indicator):
            if indicator in self.selected_indicators:
                self.selected_indicators.remove(indicator)
            else:
                self.selected_indicators.add(indicator)
            
            # Update the StringVar with the current selection
            self.scan_indicators.set(", ".join(self.selected_indicators))
            
            # Update the menubutton display text
            self.update_menubutton_text()

    def update_indicator_options(self, path):
            # Clear existing menu items
            self.dropdown_menu.delete(0, tk.END)
            self.selected_indicators.clear()
            
            # Get available indicators based on path attribute
            # This is where you'd analyze the path and determine available options
            available_indicators = self.get_available_indicators(path)
            
            # Add checkbuttons for each available indicator
            for indicator in available_indicators:
                self.dropdown_menu.add_checkbutton(
                    label=indicator,
                    onvalue=1, offvalue=0,
                    command=lambda ind=indicator: self.toggle_indicator(ind)
                )
            
            # Update the display text
            self.update_menubutton_text()
            return available_indicators

    def add_custom_indicator(self):
        """
        Open a dialog to add a custom indicator
        """
        # Prompt user for a new indicator name
        custom_indicator = simpledialog.askstring(
            "Add Custom Indicator", 
            "Enter a name for the new indicator:"
        )
        
        # If user provides a name and it's not empty
        if custom_indicator and custom_indicator.strip():
            # Add to dropdown menu
            self.dropdown_menu.add_checkbutton(
                label=custom_indicator, 
 
                command=lambda ind=custom_indicator: self.toggle_indicator(ind)
            )

    def update_menubutton_text(self):
            if not self.selected_indicators:
                self.indicators_menu.config(text="Select Indicators")
            elif len(self.selected_indicators) <= 2:
                self.indicators_menu.config(text=", ".join(self.selected_indicators))
            else:
                self.indicators_menu.config(text=f"{len(self.selected_indicators)} indicators selected")

    def get_available_indicators(self, path):
        available_indicators = set()  # Use a set to avoid duplicates
        if os.path.isdir(path):
            all_files=[]
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
                    if len(all_files) >= 100:
                        for i in range(0, len(all_files), 50): # change back from 500 for other stuff !!!!!!!
                                file_path = all_files[i]
                                try:
                                    dicom_data = pydicom.dcmread(file_path)
                                    if hasattr(dicom_data, 'SeriesDescription'):
                                        series_description = dicom_data.SeriesDescription
                                        if series_description:  # Ensure it's not empty
                                            available_indicators.add(series_description)
                                    else:
                                        dicom_data = pydicom.dcmread(file_path)
                                    if hasattr(dicom_data, 'PatientID'):
                                        series_description = dicom_data.PatientID
                                        if series_description:  # Ensure it's not empty
                                            available_indicators.add(series_description)
                                except Exception as e:
                                    print(f"Error reading COM file {file_path}: {e}")
                          
        return list(available_indicators)  # Convert set to list for the dropdown menu
    

    def toggle_indicators_entry(self):
            if self.use_default_indicators.get():
                self.saved_indicators = self.scan_indicators.get()
                self.indicators_menu.config(text= "No filtering. Might cause errors.")
                self.indicators_menu.configure(state="disabled")
               
            else:
                self.scan_indicators.set(getattr(self, 'saved_indicators', ''))
                self.indicators_menu.configure(state="normal")
    
    def on_key_change(self, *args):
        key = self.id_var.get()
        if key in id_dict:
            self.datasetname.set(id_dict[key]['body_part'])
            self.update_configurations(None)
        else: self.datasetname.set(f"The Dataset with ID {key} either does not exist.")

    def on_namechange(self, *args):
        name = self.datasetname.get()
        matching_id = next((key for key, value in id_dict.items() if value['body_part'] == name), None)
        if matching_id:
            self.id_var.set(matching_id)
            self.update_configurations(None)
    
    def drop_input_file(self, event, target_entry):
        """Handle drag and drop file input"""
        file_path = event.data.strip('{}')
        target_entry.delete(0,tk.END)
        target_entry.insert(0,file_path)
        self.status_var.set(f"Input file set to: {os.path.basename(file_path)}")
        self.update_indicator_options(file_path)

    def drop_output_files(self, event, target_entry):
        """Handle drag and drop file input"""
        file_path = event.data.strip('{}')
        target_entry.delete(0,tk.END)
        target_entry.insert(0,file_path)
        self.status_var.set(f"Output file set to: {os.path.basename(file_path)}")

    def browse_input(self):
        """Open file dialog to select input file or folder"""
        # Ask the user if they want to select a file or a folder
        choice = filedialog.askopenfilename(
            title="Select Folder",
            filetypes=(("All Files", "*.*")),
            initialdir=os.getcwd()
        )
        
        # If the user cancels the dialog, return early
        if not choice:
            return
        
        # Check if the selected path is a file or a folder
        if os.path.isfile(choice):
            self.input_path.set(choice)
            self.status_var.set(f"Input file set to: {os.path.basename(choice)}")
        elif os.path.isdir(choice):
            self.input_path.set(choice)
            self.status_var.set(f"Input folder set to: {os.path.basename(choice)}")
        else:
            self.status_var.set("Invalid selection. Please select a valid file or folder.")

    def browse_input_path(self):
        """Open directory dialog to select STL output path"""
        dir_path = filedialog.askdirectory(title="Select Input Directory")
        if dir_path:
            self.input_path.set(dir_path)
            self.status_var.set(f"Input directory set to: {os.path.basename(dir_path)}")
            self.update_indicator_options(dir_path)


    def browse_stl_output(self):
        """Open directory dialog to select STL output path"""
        dir_path = filedialog.askdirectory(title="Select STL Output Directory")
        if dir_path:
            self.stl_output_path.set(dir_path)
            self.status_var.set(f"STL output directory set to: {os.path.basename(dir_path)}")

    def browse_labelmap_output(self):
        """Open directory dialog to select labelmap output path"""
        dir_path = filedialog.askdirectory(title="Select Labelmap Output Directory")
        if dir_path:
            self.labelmap_output_path.set(dir_path)
            self.status_var.set(f"Labelmap output directory set to: {os.path.basename(dir_path)}")

    def select_all_folds(self):
        """Select all folds in the fold checkboxes"""
        for var, i in self.folds_checkbuttons:
            var.set(True)

    def update_configurations(self, event):
        """Update the available configurations based on the selected ID"""
        selected_id = self.id_var.get()
        if selected_id in id_dict:
            configurations = id_dict[selected_id]['configurations']
            self.config_combobox['values'] = configurations
            self.config_combobox.set(configurations[0])  # Set default configuration
             
    
    def clear_fields(self):
        """Clear all input fields"""
        self.input_path.set("")
        self.stl_output_path.set("")
        self.labelmap_output_path.set("")
        self.scan_indicators.set("")
        self.use_default_indicators.set(False)
        self.id_var.set("")
        self.config_var.set("3d_fullres")
        self.folds_var.set("0,1,2,3,4")
        self.status_var.set("Fields cleared")

    def update_progress(self, value, message=None):
        """Update progress bar and status message"""
        self.progress_var.set(value)
        if message:
            self.status_var.set(message)
        self.root.update_idletasks()  # Update GUI

    def process_data(self, params):
        """Process the input data using the parameters from the GUI"""
        try:
            # Convert string path to Path object if needed
            input_path = Path(params["Input Path"]) if not isinstance(params["Input Path"], Path) else params["Input Path"]
            stl_output_path = Path(params["STL Output Path"]) if params["STL Output Path"] else Path(input_path.parent) / "stl"
            labelmap_output_path = Path(params["Labelmap Output Path"]) if params["Labelmap Output Path"] else Path(input_path.parent) / 'label'
            selected_id = params["ID"]
            prefix = id_dict[selected_id]['prefix'] if selected_id in id_dict else ""
            suffix = id_dict[selected_id]['suffix'] if selected_id in id_dict else ""
            use_default = params["Use Default Indicators"]
            scan_indicators= None if use_default else params["Scan Indicators"]
            configuration = params["Configuration"]
            split = params['Split']
            multiple = params['Multiple']
            lower = params['z-lower']
            upper = params['z-upper']
            cut_z = params['cut_z']

            # Ensure output directories exist
            os.makedirs(stl_output_path, exist_ok=True)
            os.makedirs(labelmap_output_path, exist_ok=True)
            #Step 0: Convert input directory into single folder if multiple selected
            if multiple: 
                root_dir= input_path
                target_dir= Path(input_path.parent)/ 'merged'
                os.makedirs(target_dir, exist_ok=True)
                for root, dirs, files in os.walk((os.path.normpath(root_dir)), topdown=False):
                    for name in files: # removed dcm failsafe
                        subfolder_name= os.path.basename(root)
                        new_name= f'{subfolder_name}_{name}'
                        source_file = os.path.join(root, name)
                        target_file = os.path.join(target_dir, new_name)
                        shutil.copy2(source_file, target_file)
                    print("files merged")
                input_path= target_dir
            # Step 1: Convert DICOM to NIfTI
            self.update_progress(10, "Converting DICOM to NIfTI...")
            raw_data_to_nifti(input_path, scans_indicators=scan_indicators, use_default = use_default)
            
            # Step 2: Process NIfTI files - get files to process
            self.update_progress(30, "Renaming NIfTI files...")
            #length = len(list((Path(params["Input Path"]).iterdir())))
            interference_path = os.path.join(str(input_path.parent),r'NIFTI')
            if split:
                for folder in os.listdir(interference_path):
                    separator(os.path.join(interference_path,folder))
            if cut_z:
                try:
                    lower = int(lower)
                    upper = int(upper)
                    
                    if lower < 0:
                        messagebox.showerror("Error", "Lower bound must be non-negative")
                        return
                    
                    if lower >= upper:
                        messagebox.showerror("Error", "Upper bound must be greater than lower bound")
                        return
                        
                except ValueError:
                    messagebox.showerror("Error", "Lower and upper bounds must be integers")
                    return
            else:
                # If Z-cut is not enabled, use full range
                nii = load(self.nii_path)
                lower = 0
                upper = nii.shape[2]
            
            try:
                
                for folder in os.listdir(interference_path):
                    zcut(os.path.join(interference_path,folder),lower,upper)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during processing:\n{str(e)}")
                # Reset working directory if changed
    
            file_mapping = {}
            for i, folder in enumerate(os.listdir(interference_path)):
                nifti_renamer(os.path.join(interference_path,folder), prefix=prefix, suffix=suffix, number=i, file_mapping=file_mapping)
            json_path = os.path.join(input_path.parent, 'decoder.json')
            with open(json_path, 'w') as json_file:
                json.dump(file_mapping, json_file, indent=4)
            print(file_mapping)
        
            # Step 3: Use nnUNet predictor for segmentation
            self.update_progress(50, "Running nnUNet prediction...")
            selected_id = params["ID"]
            configuration = params["Configuration"]
            folds = params["Folds"]
            
            # Create the nnUNet predictor with the specified parameters
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
            
            # Set the model based on the ID and configuration
            predictor.initialize_from_trained_model_folder(
                join(nnUNet_results, id_dict[selected_id]['Path_to_results'][configuration]),
                use_folds=folds,
                checkpoint_name="checkpoint_final.pth",
            )
            
            # Run prediction on the input data
            self.update_progress(70, "Segmenting data...")
            predictor.predict_from_files(
                str(interference_path), 
                str(labelmap_output_path),
                save_probabilities=False,
                overwrite=False,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
            
            # Step 4: Convert segmentation to STL files
            self.update_progress(80, "Converting segmentations to STL...")
            
            # Get label files from the output directory
            process_directory(labelmap_output_path, stl_output_path, segment_params=segment_params[selected_id], file_mapping=file_mapping, split=split)
            
            
            self.update_progress(100, "Processing complete!")
            messagebox.showinfo("Success", "Data processing has been completed successfully!")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            return False

    def submit(self):
        """Process the form submission and run data processing"""
        # Validate required fields
        if not self.input_path.get()  or not self.id_var.get():
            messagebox.showerror("Error", "Input path and ID are required")
            return

        # Collect parameters
        params = {
            "Input Path": self.input_path.get(),
            "STL Output Path": self.stl_output_path.get(),
            "Labelmap Output Path": self.labelmap_output_path.get(),
            "Scan Indicators": self.selected_indicators,
            "Use Default Indicators": self.use_default_indicators.get(),
            "ID": self.id_var.get(),
            "Configuration": self.config_var.get(),
            "Folds": [i for var, i in self.folds_checkbuttons if var.get()],
            'Split' : self.split_nifti_var.get(),
            'Multiple': self.multiple_ids_var.get(),
            'z-lower': self.lower_var.get(),
            'z-upper': self.upper_var.get(),
            'cut_z': self.enable_zcut.get()
        }

        # Show summary of parameters
        result = "\n".join(f"{k}: {v}" for k, v in params.items())
        confirmation = messagebox.askyesno(
            "Confirm Parameters", 
            f"The following parameters will be used:\n\n{result}\n\nContinue with processing?"
        )
        
        if confirmation:
            self.status_var.set("Processing started...")
            self.progress_var.set(0)
            
            # Use a separate thread to avoid freezing the GUI
            # For simplicity, we're not using threading here, but in a real app you might want to
            self.process_data(params)

    def edit_segment_params(self):

        selected_id = self.id_var.get()
        if not selected_id:
            messagebox.showerror("Error", "Please select an ID or name first.")
            return

        segment_window = tk.Toplevel(self.root)
        segment_window.title("Edit Segment Params")
        segment_window.geometry("500x400")
        
        # Create a frame with scrollbar for the parameters
        canvas = tk.Canvas(segment_window)
        scrollbar = ttk.Scrollbar(segment_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the scrollbar and canvas
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store references to Entry widgets
        param_entries = {}
        
        # Add parameter entries for the selected ID
        segment_frame = ttk.LabelFrame(scrollable_frame, text=f"Segment Parameters for ID {selected_id}")
        segment_frame.pack(fill="x", expand=True, padx=10, pady=5)
        
        param_entries[selected_id] = {}
        
        row = 0
        for label_id, params in segment_params[selected_id].items():
            label_frame = ttk.LabelFrame(segment_frame, text=f"Label {label_id} ({params['label']})")
            label_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
            
            for param_name, param_value in params.items():
                if param_name != 'label' and isinstance(param_value, (int, float, str)):
                    ttk.Label(label_frame, text=f"{param_name}:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    entry = ttk.Entry(label_frame)
                    entry.insert(0, str(param_value))
                    entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
                    param_entries[selected_id][(label_id, param_name)] = entry
                    row += 1
    
    # Function to save the updated parameters
        def save_params():
            try:
                for (label_id, param_name), entry_widget in param_entries[selected_id].items():
                    value = entry_widget.get()
                    # Convert to appropriate type
                    if param_name == 'mesh_smoothing_method':
                        segment_params[selected_id][label_id][param_name] = value
                    elif param_name == 'mesh_smoothing_iterations':
                        segment_params[selected_id][label_id][param_name] = int(value)
                    else:
                        segment_params[selected_id][label_id][param_name] = float(value)
                
                messagebox.showinfo("Success", "Segment parameters updated successfully")
                segment_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid value entered: {str(e)}")
        
        # Add save and cancel buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", expand=True, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save", command=save_params).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=segment_window.destroy).pack(side="right", padx=5)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ParameterGUI(root)
    root.mainloop()
