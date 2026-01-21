import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from ttkbootstrap.scrolled import ScrolledFrame
from ttkbootstrap.tooltip import ToolTip
import os
import re
import sys
import json
import pathlib
from pathlib import Path
import numpy as np
import threading
import queue
import dicom2nifti
import pydicom 
import dicom2nifti.settings as settings 
from nnunetv2.paths import nnUNet_results, nnUNet_raw 
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerLoRA import create_lora_predictor
import subprocess
from utils.json_renamer import rename_keys
import nibabel as nib
from nibabel import load, Nifti1Image, save 
from skimage import measure
from scipy import ndimage
import pyvista as pv
from stl import mesh
from multi_stl import process_directory_parallel
import shutil
from cutting import masking, zcut, cut_volume
from DICOMtoNIFTI import raw_data_to_nifti_parallel, nifti_renamer
from modifier import  stl_renamer_with_lut
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import logging
from utils.logging_tool import gui_log_output, SuppressStdout, TerminalOnlyStdout
from utils.analytics import calculate_hu_stats
with open("ids.json", "r") as ids:
    id_dict = json.load(ids)
with open("labels.json", "r") as labels:
    labels_dict = json.load(labels)
from typing import TypedDict, List, Optional, Union, Dict, Any, Set
# pyright: reportAttributeAccessIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportWildcardImportFromLibrary= false
# pyright: reportCallIssue = false 



AppParameters = TypedDict('AppParameters', {
    "Input Path": str,
    "STL Output Path": Optional[str],
    "Labelmap Output Path": Optional[str],
    "Scan Indicators": Optional[List[str]],
    "Group Filter": Optional[str],
    "Use Default Indicators": bool,
    "ID": str,
    "Configuration": str,
    "Folds": List[int],
    "Split": bool,
    "cut_enabled": bool,
    "lower_x": Optional[str],
    "upper_x": Optional[str],
    "lower_y": Optional[str],
    "upper_y": Optional[str],
    "lower_z": Optional[str],
    "upper_z": Optional[str],
    "keep_originals": bool,
    "use_percent": bool,
    "used_lps": bool,
    "use_meshrepair": bool,
    "remove_islands": bool,
    "name_only": bool,
    "nifti_input": bool,
    "run_analytics": bool
}, total=True)

# Default segment parameters -  smoothing for labelmap just removes salt-pepper noise, taubin maintains volume with  light smoothing-params to remove some steps\artifacts 
default_segment_params = {
    'smoothing': 1.0, 
    'mesh_smoothing_method': 'taubin', 
    'mesh_smoothing_iterations': 150, 
    'mesh_smoothing_factor': 0.1
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

class ProgressEvent:
    def __init__(self, value, message=None, error=None, completed=False):
        self.value = value
        self.message = message
        self.error = error
        self.completed = completed


class ParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parameter Input GUI for nnUNet segmentation")
        self.root.geometry("1300x750")
        self.style = tb.Style(theme='superhero') 
         # Create a queue for thread communication
        self.progress_queue = queue.Queue()
        
        # Flag to track if processing is currently running
        self.processing_running = False
        
        # Start progress monitoring
        self.poll_progress_queue()

        # Create main frame with padding
        main_frame = tb.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)

        path_frame= tb.LabelFrame(main_frame, text="Input/Output paths", padding = "10 ", bootstyle= INFO)
        path_frame.grid(row=0, column=0, columnspan=5, sticky = EW, pady=(0,10))
        path_frame.columnconfigure(1, weight=1)
      
      
       
        #Input path
        tb.Label(path_frame, text="Input Path:", bootstyle= PRIMARY).grid(row=0, column=0, sticky=W, pady=5, padx= 5)
        self.input_path = tk.StringVar()
        input_frame = tb.Frame(path_frame)
        input_frame.grid(row=0, column=1, sticky=EW, pady=5, padx=5)
        input_frame.columnconfigure(0, weight=1)
        self.input_entry = tb.Entry(input_frame, textvariable=self.input_path, width=60)
        self.input_entry.grid(row=0, column=0, sticky = EW)
        self.input_entry.drop_target_register(DND_FILES)
        self.input_entry.dnd_bind('<<Drop>>', lambda e: self.drop_input_file(e, self.input_entry)) 
        input_button = tb.Button(input_frame, text="Browse", command=self.browse_input_path, bootstyle = "primary")
        input_button.grid(row=0, column=1,  padx=(5, 0))

        #Output paths
        tb.Label(path_frame, text="STL Output Path:").grid(row=1, column=0, sticky=W, pady=5, padx=5 )
        self.stl_output_path = tk.StringVar()
        stl_output_frame = tb.Frame(path_frame)
        stl_output_frame.grid(row=1, column=1, sticky=EW, pady=5, padx=5)
        stl_output_frame.columnconfigure(0, weight=1)
        self.stl_output_entry = tb.Entry(stl_output_frame, textvariable=self.stl_output_path, width=60)
        self.stl_output_entry.grid(row=0, column = 0, sticky = EW)
        self.stl_output_entry.drop_target_register(DND_FILES)
        self.stl_output_entry.dnd_bind('<<Drop>>', lambda e: self.drop_output_files(e, self.stl_output_entry))
        stl_output_button = tb.Button(stl_output_frame, text="Browse", command=self.browse_stl_output, bootstyle= "secondary")
        stl_output_button.grid(row=0, column = 1, padx= (5,0 ))

        tb.Label(path_frame, text="nnUNet Labelmap Output Path:").grid(row=2, column=0, sticky=W, pady=(0, 10))
        self.labelmap_output_path = tk.StringVar()
        labelmap_output_frame = tb.Frame(path_frame)
        labelmap_output_frame.grid(row=2, column=1, sticky=EW, pady=5, padx= 5)
        labelmap_output_frame.columnconfigure(0, weight=1)
        self.labelmap_output_entry = ttk.Entry(labelmap_output_frame, textvariable=self.labelmap_output_path, width=60)
        self.labelmap_output_entry.grid(row=0, column =0, sticky = EW)
        self.labelmap_output_entry.drop_target_register(DND_FILES)
        self.labelmap_output_entry.dnd_bind('<<Drop>>',  lambda e: self.drop_output_files(e, self.labelmap_output_entry))
        labelmap_output_button = ttk.Button(labelmap_output_frame, text="Browse", command=self.browse_labelmap_output, bootstyle= "secondary")
        labelmap_output_button.grid(row=0, column= 1, padx= (5,0))

        # Row 3: Scan Indicators (as set of strings)
        indicator_frame = tb.LabelFrame(main_frame, text= "Filter out specific scan", padding = "10", bootstyle= WARNING)
        indicator_frame.grid(row=2, column=0, columnspan=5, sticky=EW, pady=(0,10))
        indicator_frame.columnconfigure(0, weight=1)


        tb.Label(indicator_frame, text="Scan Indicators:").grid(row=0, column =0, sticky = W, pady=5, padx = 5)

        # Create a StringVar to store the selected indicators
        self.scan_indicators = tk.StringVar()

        # Create a set to store the selected indicators
        self.selected_indicators = set()

        indicator_controls_frame = tb.Frame(indicator_frame)
        indicator_controls_frame.grid(row=0, column=1, sticky=EW, padx=5, pady=5)
       
        indicator_controls_frame.columnconfigure(0, weight=1) 
        indicator_controls_frame.columnconfigure(3, weight=1)

        self.indicators_menu = ttk.Menubutton(indicator_controls_frame, text="Select Indicators", width=60, direction='flush')
        self.indicators_menu.grid(row=0, column=0, sticky=EW, padx=(0, 5))
       
        self.dropdown_menu = tk.Menu(self.indicators_menu, tearoff=0)
        self.indicators_menu["menu"] = self.dropdown_menu

        # Custom Indicator Button
        self.add_custom_button = tb.Button(
            indicator_controls_frame, 
            text="+", 
            width=3, 
            command=self.add_custom_indicator, bootstyle= "info", ) 
        
        
        self.add_custom_button.grid(row=0, column=1, padx=(5, 5))

        tb.Label(indicator_controls_frame, text="Group:").grid(row=0, column=2, padx=(10, 2), sticky=W)
        self.group_filter_var = tk.StringVar()
        self.group_filter_entry = tb.Entry(indicator_controls_frame, textvariable=self.group_filter_var, width=20)
        self.group_filter_entry.grid(row=0, column=3, sticky=EW, padx=(0, 5))
        self.group_filter_var.trace_add('write', self.check_filter_activity)
        # -------------------------

        # Checkbox for using default indicators
        self.use_default_indicators = tk.BooleanVar(value=True) # Start as True
        self.default_indicators_check = tb.Checkbutton(
            indicator_controls_frame, 
            text="Don't filter", 
            variable=self.use_default_indicators,
            command=self.toggle_indicators_entry, bootstyle= "warning", 
        )
        self.default_indicators_check.grid(row=0, column=4, padx=(10, 10), sticky=W)

       
        

        # Row 4: Select ID and Configurations

        config_group_frame = tb.LabelFrame(main_frame, text="Details for the segmentation", padding = "10", bootstyle= INFO)
        config_group_frame.grid(row=3, column=0, columnspan=1, sticky=EW, pady=(0, 5)) # Use grid in main_frame
        config_group_frame.columnconfigure(1, weight=1)



        tb.Label(config_group_frame, text="ID:").grid(row=0, column=0, sticky=W, pady=5, padx = 5)
        self.id_var = tk.StringVar()
        self.id_var.trace_add('write', self.on_key_change)
        id_combobox = ttk.Combobox(config_group_frame, textvariable=self.id_var, width=40)
        id_combobox['values'] = list(id_dict.keys())
        id_combobox.grid(row=0, column=1, sticky=W, pady=5, padx = 5)
        id_combobox.bind("<<ComboboxSelected>>", self.update_configurations)
        
        tb.Label(config_group_frame, text="Dataset name:").grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        self.datasetname = tk.StringVar()
        self.datasetname.trace_add('write', self.on_namechange)
        self.datasetname.set("Select Name.")
        name_combobox= tb.Combobox(config_group_frame, textvariable= self.datasetname, width= 40)
        name_combobox['values'] = [id_dict[id_key]['body_part'] for id_key in list(id_dict.keys())]
        name_combobox.grid(row=1, column=1, sticky=W, pady=5, padx =  5)
        name_combobox.bind("<<ComboboxSelected>>", self.on_namechange)

        

        tb.Label(config_group_frame, text="Configuration:").grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        self.config_var = tk.StringVar()
        self.config_combobox = tb.Combobox(config_group_frame, textvariable=self.config_var, width=40)
        self.config_combobox.grid(row=2, column=1, sticky=tk.W, padx = 5, pady = 5)

        # Row 5: Folds (check boxes)
        ttk.Label(config_group_frame, text="Folds:").grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        self.folds_frame = tb.Frame(config_group_frame)
        self.folds_frame.grid(row=3, column=1, sticky=EW, pady=10, padx = 5)
        self.folds_var = tk.StringVar(value="0,1,2,3,4")  # Default value of 5 folds
        self.folds_checkbuttons = []
        for i in range(5):
            var = tk.BooleanVar(value=True)  
            check = tb.Checkbutton(self.folds_frame, text=f"Fold {i}", variable=var)
            check.grid(row=0, column=i, padx=5, pady=5)
            self.folds_checkbuttons.append((var, i))

        self.select_all_button = ttk.Button(self.folds_frame, text="Select All", command=self.select_all_folds)
        self.select_all_button.grid(row=1, column=0, columnspan=2, padx=15, pady=5)
        self.deselect_all_button = ttk.Button(self.folds_frame, text= "Deselect all", command=self.deselect_all_folds, style= WARNING )
        self.deselect_all_button.grid(row=1, column=2, columnspan=2, padx=15, pady=5 )
      
       #Processing
    
        preprocessing_frame = tb.LabelFrame(main_frame, text="Processing Options", padding = "10", bootstyle= INFO)
        preprocessing_frame.grid(row=3, column=1, columnspan=3, sticky=EW, pady=(0, 5), padx=5) # Use grid in main_frame
        preprocessing_frame.columnconfigure(1, weight=1)




        self.split_nifti_var = tk.BooleanVar(value=False)
        self.split_nifti_check = tb.Checkbutton(preprocessing_frame, text="Split NIFTIs into left and right along x axis", variable=self.split_nifti_var)
        self.split_nifti_check.grid(row=0, column=0, columnspan=3,   sticky=tk.W, pady= (0,10) )


        self.enable_cut = tk.BooleanVar(value=False)
        self.keep_originals = tk.BooleanVar(value=False)
        self.use_percent = tk.BooleanVar(value=False)
        self.used_lps = tk.BooleanVar(value=False)
        self.lower_x = tk.StringVar(value="")
        self.upper_x = tk.StringVar(value="")
        self.lower_y = tk.StringVar(value="")
        self.upper_y = tk.StringVar(value="")
        self.lower_z = tk.StringVar(value="")
        self.upper_z = tk.StringVar(value="")

        self.cut_entries = []


        enable_checkbox = tb.Checkbutton(
            preprocessing_frame, 
            text="Cut Volume (Crop)",
            variable=self.enable_cut,
            command=self.toggle_cut_inputs
        )
        enable_checkbox.grid(row=5, column=0, sticky="w", pady=5)
        
        help_text = (
            "Cropping Guide for (RAS+ System):\n"
            "• X: Lower = Left, Upper = Right\n"
            "• Y: Lower = Back, Upper = Front\n"
            "• Z: Lower = Feet, Upper = Head\n"
            "This is different from the DICOMS which are LPS. Click the button for LPS.\n"
            "Then add bounds as seen in the Viewer, here the right patient side (left of image) will have the lower values. \n"
            "The script handles the conversion to RAS+ for cutting. \n"
            "So be aware of what system you used and what you cut by doing so.\n"
            "If using %, 0 is start of image, 100 is end."
        )
        ToolTip(enable_checkbox, text=help_text, delay=200, bootstyle="info", position= "top" )
        self.keep_originals_checkbox = tb.Checkbutton(
            preprocessing_frame, 
            text="Keep originals", 
            variable=self.keep_originals, 
            state="disabled"
        )
        self.keep_originals_checkbox.grid(row=6, column=0, columnspan=1, sticky="w", pady=0, padx=(40,10))

        self.percent_checkbox = tb.Checkbutton(
            preprocessing_frame, 
            text="Use Percentages", 
            variable=self.use_percent, 
            state="disabled" 
            
        )
        self.percent_checkbox.grid(row=7, column=0, columnspan=1, sticky="w", pady=0, padx=(40,10))

        self.LPS_checkbox = tb.Checkbutton(
            preprocessing_frame, 
            text="Use LPS Coordinates to cut.", 
            variable=self.used_lps, 
            state="disabled" 
            
        )
        self.LPS_checkbox.grid(row=8, column=0, columnspan=1, sticky="w", pady=0, padx=(40,10))
        ToolTip(self.LPS_checkbox, text="If you used e.g. MicroDicom to check the images. They are in LPS so X and Y are flipped for our case. It handles the conversion. Ignore the other help statement as now X and Y are flipped.", delay=200, bootstyle="info", position="bottom")

        # --- Column Headers (Row 1) ---
        # Visual helper to tell user which column is which
        ttk.Label(preprocessing_frame, text="Lower Bound").grid(row=5, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(preprocessing_frame, text="Upper Bound").grid(row=5, column=1, padx=15, pady=2, )

        # --- Axis Rows (Rows 2, 3, 4) ---
        axes_settings = [
            ("X-Axis:", self.lower_x, self.upper_x, 2),
            ("Y-Axis:", self.lower_y, self.upper_y, 3),
            ("Z-Axis:", self.lower_z, self.upper_z, 4),
        ]

        for label_text, low_var, up_var, row_idx in axes_settings:
            # Axis Label
            ttk.Label(preprocessing_frame, text=label_text).grid(row=row_idx+4, column=0, sticky="e", padx=(250,10), pady=2)
            
            # Lower Entry
            entry_l = tb.Entry(preprocessing_frame, textvariable=low_var, width=10, state="disabled")
            entry_l.grid(row=row_idx+4, column=1, padx=5, pady=2, sticky="w")
            self.cut_entries.append(entry_l)
            
            # Upper Entry
            entry_u = tb.Entry(preprocessing_frame, textvariable=up_var, width=10, state="disabled")
            entry_u.grid(row=row_idx+4, column=1, padx=5, pady=2)
            self.cut_entries.append(entry_u)

   
        self.just_name = tk.BooleanVar(value=True)
        self.just_name_check =tb.Checkbutton(preprocessing_frame, text="Just use the name for the folders/names, safe if IDs are messy with timescodes etc.", variable=self.just_name)
        self.just_name_check.grid(row=2, column=0, columnspan=1, sticky=tk.W, pady=(0, 10))

 
        self.meshfix_var = tk.BooleanVar(value=True)
        self.meshfix_check = tb.Checkbutton(preprocessing_frame, text= "Appply Pymesh meshrepair (caps open stls as well)",  variable=self.meshfix_var, command=self.on_meshfix_toggle)
        self.meshfix_check.grid(row=3, column=0, columnspan=1, sticky = tk.W, pady=(0,10))

    
        self.islands_var = tk.BooleanVar(value=True)
        self.islands_check = tb.Checkbutton(preprocessing_frame, text= "Remove all but the largest element from stl",  variable=self.islands_var)
        self.islands_check.grid(row=4, column=0, columnspan=1, sticky = tk.W, pady=(0,10))

        self.analytics_var = tk.BooleanVar(value=False)
        self.analytics_check = tb.Checkbutton(preprocessing_frame, text= "Run HU analytics",  variable=self.analytics_var)
        self.analytics_check.grid(row=15, column=0, columnspan=1, sticky = tk.W, pady=(0,10))


        #Option for Loading NIFTIs
        config_input_frame = tb.LabelFrame(main_frame, text= "Input options", padding = "10", bootstyle= WARNING)
        config_input_frame.grid(row=1, column=0, columnspan=6, sticky=EW, pady=(0,10)) # Use grid in main_frame
        config_input_frame.columnconfigure(0, weight=1)

        self.input_nifti = tk.BooleanVar(value=False)
        self.input_nifti_check = tb.Checkbutton(config_input_frame, text= "NIFTIs are used as input, disables conversion. Click Keep originals for cropping otherwise it is destructive.",  variable=self.input_nifti)
        self.input_nifti_check.grid(row=0, column=0, columnspan=5, sticky = tk.W, pady=(0,10))

        # Button frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))
        submit_button = tb.Button(button_frame, text="Submit", command=self.submit, bootstyle= SUCCESS)
        submit_button.pack(side=tk.LEFT, padx=(0, 10))
        clear_button = tb.Button(button_frame, text="Clear", command=self.clear_fields, bootstyle = WARNING)
        clear_button.pack(side=tk.LEFT)

        # Progress bar for displaying operation progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tb.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0)) #type: ignore
        
        # Segment parameters editing window
        self.segment_params_button = tb.Button(button_frame, text="Edit Segment Params", command=self.edit_segment_params)
        self.segment_params_button.pack(side=tk.LEFT, padx = (10,0))

        # Configure grid to expand with window resizing
        main_frame.columnconfigure(1, weight=1)

        # Status bar at bottom of window
        self.status_var = tk.StringVar()
        status_bar = tb.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, bootstyle = INFO)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

        self.input_path.trace_add('write', self.schedule_indicator_scanning)
        self.check_filter_activity()

    def schedule_indicator_scanning(self, *args):
    
        path = self.input_path.get()
        if os.path.isdir(path):
            # Reset indicator menu
            self.dropdown_menu.delete(0, tk.END)
            self.selected_indicators.clear()
            self.update_menubutton_text()
            
            # Show loading indicator in menu
            self.indicators_menu.config(text="Scanning for indicators...")
            
            # Start scanning thread
            scan_thread = threading.Thread(target=self.scan_indicators_thread, args=(path,))
            scan_thread.daemon = True
            scan_thread.start()

    def scan_indicators_thread(self, path):
        """Thread function to scan for indicators"""
        try:
            indicators = set()
            
            # Walk through directory and find DICOM files
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Try to read as DICOM
                        dicom_data = pydicom.dcmread(file_path, stop_before_pixels=True)
                        
                        # Check for SeriesDescription
                        if hasattr(dicom_data, 'SeriesDescription'):
                            desc = dicom_data.SeriesDescription
                            if desc:
                                indicators.add(desc)
                        
                        # Check for PatientID
                        elif hasattr(dicom_data, 'PatientID'):
                            pid = dicom_data.PatientID
                            if pid:
                                indicators.add(pid)
                        
                        # Break after finding a few DICOM files with indicators
                    except Exception:
                        continue  # Skip non-DICOM files
            
            
            # Update UI in main thread
            self.root.after(0, lambda: self.update_indicators_options(sorted(indicators)))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error scanning indicators: {str(e)}"))

    def update_indicators_options(self, indicators):
        """Update the indicators menu with the scanned indicators"""
        # Clear the menu and selected indicators
        self.dropdown_menu.delete(0, tk.END)
        self.selected_indicators.clear()
        
        # Add checkbuttons for each available indicator
        for indicator in indicators:
            var = tk.BooleanVar(value=False)
            self.dropdown_menu.add_checkbutton(
                label=indicator,
                variable=var,
                command=lambda ind=indicator: self.toggle_indicator(ind)
            )
    
    # Update the menubutton text
        self.update_menubutton_text()
        
        # Update status
        if indicators:
            self.indicators_menu.config(text="Select Indicators")
            self.status_var.set(f"Found {len(indicators)} indicators")
        else:
            self.indicators_menu.config(text="No indicators found")
            self.status_var.set("No indicators found in the selected directory")

    def on_meshfix_toggle(self):
        if not self.meshfix_var.get():
            # If meshfix is unchecked, uncheck and disable islands
            self.islands_var.set(False)
            self.islands_check.configure(state='disabled')
        else:
            # If meshfix is checked, enable islands checkbox
            self.islands_check.configure(state='normal')


    def toggle_indicator(self, indicator):
            if indicator in self.selected_indicators:
                self.selected_indicators.remove(indicator)
            else:
                self.selected_indicators.add(indicator)
            
            # Update the StringVar with the current selection
            self.scan_indicators.set(", ".join(self.selected_indicators))
            
            # Update the menubutton display text
            self.update_menubutton_text()
            self.check_filter_activity()
    def toggle_cut_inputs(self):
        """Enable or disable all cutting inputs based on the main checkbox."""
        # Determine state based on the checkbox
        state = "normal" if self.enable_cut.get() else "disabled"
        
        # Toggle the 'Keep Originals' checkbox
        self.keep_originals_checkbox.configure(state=state)
        self.percent_checkbox.configure(state=state)
        self.LPS_checkbox.configure(state=state)
        # Toggle all Entry widgets
        for entry in self.cut_entries:
            entry.configure(state=state)

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


    def toggle_indicators_entry(self):
            if self.use_default_indicators.get():
                self.saved_indicators = self.scan_indicators.get()
                self.indicators_menu.config(text= "No filtering. Will convert all viable scans.")
                self.indicators_menu.configure(state="disabled")
                self.group_filter_entry.configure(state="disabled")
               
            else:
                self.scan_indicators.set(getattr(self, 'saved_indicators', ''))
                self.indicators_menu.configure(state="normal")
                self.group_filter_entry.configure(state="normal")


    def check_filter_activity(self, *args):
        """Auto-toggle 'Don't filter' based on filter inputs."""
        no_indicators = not self.selected_indicators
        no_group_filter = not self.group_filter_var.get().strip()
        
        if no_indicators and no_group_filter:
            # If no filters are active, check "Don't filter"
            if not self.use_default_indicators.get():
                self.use_default_indicators.set(True)
                self.toggle_indicators_entry()
        else:
            # If any filter is active, uncheck "Don't filter"
            if self.use_default_indicators.get():
                self.use_default_indicators.set(False)
                self.toggle_indicators_entry()
    # --------------------
    
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

    def toggle_zcut_inputs(self):
        if self.enable_zcut.get():
            self.keep_originals_checkbox["state"] = "normal"
            self.lower_entry["state"] = "normal"
            self.upper_entry["state"] = "normal"
        else:
            self.keep_originals.set(False)
            self.keep_originals_checkbox["state"] = "disabled"
            self.lower_entry["state"] = "disabled"
            self.upper_entry["state"] = "disabled"
    
    def drop_input_file(self, event, target_entry):
        """Handle drag and drop file input"""
        file_path = event.data.strip('{}')
        target_entry.delete(0,tk.END)
        target_entry.insert(0,file_path)
        self.status_var.set(f"Input file set to: {os.path.basename(file_path)}")
        #self.update_indicator_options(file_path)

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
            filetypes=(("All Files", "*.*")), #type: ignore
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
            #self.update_indicator_options(dir_path)


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
    def deselect_all_folds(self):
        for var, i in self.folds_checkbuttons:
            var.set(False)

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
        self.group_filter_var.set("")
        self.use_default_indicators.set(False)
        self.id_var.set("")
        self.config_var.set("3d_fullres")
        self.folds_var.set("")
        self.status_var.set("Fields cleared")

    def update_progress(self, value, message=None):
        """Update progress bar and status message"""
        self.progress_var.set(value)
        if message:
            self.status_var.set(message)
        self.root.update_idletasks()  # Update GUI

    
 

    @gui_log_output(get_log_dir_from_args=lambda s, params: Path(params["Input Path"]).parent / "logs")
    def process_data(self, params : AppParameters):
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
            group_filter = None if use_default else params["Group Filter"]
            configuration = params["Configuration"]
            split = params['Split']
            just_name = params['name_only']
            cut_enabled = params['cut_enabled']
            keep_originals = params["keep_originals"]
            use_percent = params["use_percent"]
            meshrepair = params["use_meshrepair"]
            remove_islands = params["remove_islands"]
            nifti_input = params["nifti_input"]
            analytics = params["run_analytics"]
            used_lps = "LPS" if params["used_lps"] else "RAS"
            # Ensure output directories exist
            os.makedirs(stl_output_path, exist_ok=True)
            os.makedirs(labelmap_output_path, exist_ok=True)
            if not nifti_input: 
                # Step 1: Convert DICOM to NIfTI 
                self.progress_queue.put(ProgressEvent(10, "Converting DICOM to NIfTI..."))
                
                raw_data_to_nifti_parallel(input_path, scans_indicators=scan_indicators, group_filter=group_filter, use_default=use_default, max_workers=14, use_only_name=just_name, max_workers_dicom=32) #change back later to 12 
                inference_path = os.path.join(str(input_path.parent),r'NIFTI')
            # Step 2: Process NIfTI files - get files to process
            
            else: 
                self.progress_queue.put(ProgressEvent(10, "Starting with NIfTI..."))
                inference_path = input_path
            
            
           
            
                
            if cut_enabled:
                self.progress_queue.put(ProgressEvent(30, "Processing NIfTI files"))
                try:
                    # Helper to convert GUI string inputs to int or None
                    def get_val(val):
                        return int(val) if val and str(val).strip() != "" else None

                    # 1. Collect inputs into variables from params
                    lx, ly, lz = get_val(params['lower_x']), get_val(params['lower_y']), get_val(params['lower_z'])
                    ux, uy, uz = get_val(params['upper_x']), get_val(params['upper_y']), get_val(params['upper_z'])

                    # 2. Form the tuples (x, y, z)
                    lower_bounds = (lx, ly, lz)
                    upper_bounds = (ux, uy, uz)

                    # 3. Validate bounds for all axes
                    axes_labels = ['x', 'y', 'z']
                    for i in range(3):
                        l = lower_bounds[i]
                        u = upper_bounds[i]

                        if l is not None and l < 0:
                            messagebox.showerror("Error", f"Lower bound for {axes_labels[i]}-axis must be non-negative")
                            return False # Return False to stop processing

                        if l is not None and u is not None:
                            if l >= u:
                                messagebox.showerror("Error", f"Upper bound must be greater than lower bound for {axes_labels[i]}-axis")
                                return False

                    # 4. Iterate and process
                    # Make sure cut_volume is imported from your cutting module!
                    for filename in os.listdir(inference_path):
                        full_path = os.path.join(inference_path, filename)
                        if keep_originals: 
                            cut_path = Path(input_path.parent) / "NIFTI_cropped"
                            if os.path.isfile(full_path) and (filename.endswith('.nii') or filename.endswith('.nii.gz')):
                                cut_volume(
                                    nii_path=full_path,
                                    lower=lower_bounds,
                                    upper=upper_bounds,
                                    keep_original=keep_originals, 
                                    destination_dir=cut_path,
                                    localiser="cut", percents_given=use_percent, input_type=used_lps
                                )
                        else: 
                                cut_volume(
                                    nii_path=full_path,
                                    lower=lower_bounds,
                                    upper=upper_bounds,
                                    keep_original=keep_originals, 
                                    destination_dir=inference_path,
                                    localiser="cut", percents_given=use_percent, input_type=used_lps
                                )
                    if keep_originals: 
                        inference_path = cut_path
                            
                except ValueError:
                    messagebox.showerror("Error", "Please ensure all coordinates are valid integers.")
                    return False
                except Exception as e:
                    messagebox.showerror("Error", f"The NIFTIs cannot be cut. Details: {str(e)}")
                    return False
           
   
            self.progress_queue.put(ProgressEvent(40, "Renaming NIfTI files..."))
            file_mapping = {}
            for i, folder in enumerate(os.listdir(inference_path)):
                nifti_renamer(os.path.join(inference_path,folder), prefix=prefix, suffix=suffix, number=i, file_mapping=file_mapping)
            json_path = os.path.join(input_path.parent, 'decoder.json')
            with open(json_path, 'w') as json_file:
                json.dump(file_mapping, json_file, indent=4)
            print(file_mapping)
        
            # Step 3: Use nnUNet predictor for segmentation
            self.progress_queue.put(ProgressEvent(50, "Setting up nnUNet predictor..."))
            selected_id = params["ID"]
            configuration = params["Configuration"]
            folds = params["Folds"]
            print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
            if selected_id == "118": 
                predictor = create_lora_predictor(predictor)
                print("Using LoRA model for segmentation.")

            if selected_id in ("117", "118"):
                lowres_path = Path(input_path.parent) / 'label_lowres'
                os.makedirs(Path(input_path.parent) / 'label_lowres', exist_ok=True)
                print(f"Running cascade segemntation for ID {selected_id}" )
                predictor.initialize_from_trained_model_folder(
                join(nnUNet_results, id_dict[selected_id]['Path_to_results']['3d_lowres']),
                use_folds=(0,1,2,3,4),
                checkpoint_name="checkpoint_final.pth",
            )
                
                # Run prediction on the input data
            
                self.progress_queue.put(ProgressEvent(60, "Running segmentation for lowres..."))
                print("Starting nnU-Net prediction. Status updates will be suppressed from the log.")
                #with TerminalOnlyStdout():
                predictor.predict_from_files(
                    str(inference_path), 
                    str(lowres_path),
                    save_probabilities=False,
                    overwrite=False,
                    num_processes_preprocessing=4,
                    num_processes_segmentation_export=4,
                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
                print("Done with lowres")

                if selected_id == "118":
                    predictor.initialize_from_trained_model_folder(
                    join(nnUNet_results, id_dict[selected_id]['Path_to_results']['3d_cascade_fullres']),
                    use_folds=(0,1,2,3),
                    checkpoint_name="checkpoint_final.pth",)
                else: 
                     predictor.initialize_from_trained_model_folder(
                    join(nnUNet_results, id_dict[selected_id]['Path_to_results']['3d_cascade_fullres']),
                    use_folds=(0,),
                    checkpoint_name="checkpoint_final.pth",)
    
                # Run prediction on the input data
            
                self.progress_queue.put(ProgressEvent(60, "Running segmentation for fullres cascade..."))
                print("Starting nnU-Net prediction. Status updates will be suppressed from the log.")
                with TerminalOnlyStdout():
                    predictor.predict_from_files(
                        str(inference_path), 
                        str(labelmap_output_path),
                        save_probabilities=False,
                        overwrite=False,
                        num_processes_preprocessing=4,
                        num_processes_segmentation_export=4,
                        folder_with_segs_from_prev_stage=str(lowres_path), num_parts=1, part_id=0)
                    

                print("nnUNet segmentation done.")
           
            else:
            # Set the model based on the ID and configuration
                predictor.initialize_from_trained_model_folder(
                    join(nnUNet_results, id_dict[selected_id]['Path_to_results'][configuration]),
                    use_folds=folds,
                    checkpoint_name="checkpoint_final.pth",
                )
                
                # Run prediction on the input data
            
                self.progress_queue.put(ProgressEvent(60, "Running segmentation..."))
                print("Starting nnU-Net prediction. Status updates will be suppressed from the log.")
                with TerminalOnlyStdout():
                    predictor.predict_from_files(
                        str(inference_path), 
                        str(labelmap_output_path),
                        save_probabilities=False,
                        overwrite=False,
                        num_processes_preprocessing=2,
                        num_processes_segmentation_export=2,
                        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
                print("nnUNet segmentation done.")

            #Masking after segmentation, should not cause problems in the segmentation is faster and background is 0 for every file 
            
            if split:
                for folder in os.listdir(labelmap_output_path):
                    if folder.endswith(".nii.gz"):
                        masking(os.path.join(labelmap_output_path,folder))

            # Step 4: Convert segmentation to STL files
            self.progress_queue.put(ProgressEvent(80, "Converting segmentations to STL..."))
            print("Starting with batched stl conversion.")
            # Get label files from the output directory
            stl_metadata_path = Path(input_path.parent) / "stl_metadata.json"
            process_directory_parallel(labelmap_output_path, stl_output_path, segment_params=segment_params[selected_id], split=split, use_pymeshfix=meshrepair, remove_islands=remove_islands, max_workers=10, stl_metadata_path=stl_metadata_path)
            
            self.progress_queue.put(ProgressEvent(90, "STL names back to original names..."))
            reverse_mapping_number = stl_renamer_with_lut(stl_output_path=stl_output_path, file_mapping=file_mapping)
            stl_metadata_path = rename_keys(stl_metadata_path, stl_metadata_path, reverse_mapping_number)
            cleaned_metapath = str(Path(stl_metadata_path).resolve())
            if analytics:
                self.progress_queue.put(ProgressEvent(95, "Running HU analytics..."))
                stats_dir = Path(input_path.parent) / "HU_Analytics"
                calculate_hu_stats(inference_path, labelmap_output_path, file_mapping, stats_dir, id= selected_id, labels_dict= labels_dict, number_to_name_dict= reverse_mapping_number, stl_metadata_path=cleaned_metapath)

            
            
            python_exe = sys.executable
            cmd = [python_exe, "-m", "streamlit", "run", "utils/streamlit_dbscan.py", "--",  cleaned_metapath] 
            subprocess.run(cmd)
            
            self.progress_queue.put(ProgressEvent(100, "Processing complete!", completed=True))
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            return False
        
    def poll_progress_queue(self):
        """Check progress queue for updates and update UI accordingly"""
        try:
            while True:
                event = self.progress_queue.get_nowait()
                if event.error:
                    self.processing_running = False
                    messagebox.showerror("Error", f"An error occurred: {event.error}")
                    self.status_var.set(f"Error: {event.error}")
                elif event.completed:
                    self.processing_running = False
                    messagebox.showinfo("Success", "Processing completed successfully!")
                    self.status_var.set("Processing complete!")
                else:
                    if event.value is not None:
                        self.progress_var.set(event.value)
                    if event.message:
                        self.status_var.set(event.message)
                
                self.progress_queue.task_done()
        except queue.Empty:
            pass
        
        # Reschedule poll after 100ms
        self.root.after(100, self.poll_progress_queue)
    

    def submit(self):
        """Process the form submission and run data processing"""
        # Validate required fields

         # Check if processing is already running
        if self.processing_running:
            # Ask the user if they want to force a restart
            confirm_override = messagebox.askyesno(
                "Processing in Progress", 
                "An operation is currently running.\n\n"
                "Do you want to force a restart? (The previous background process will continue until completion, but the UI will track the new one)."
            )
            
            if confirm_override:
                # Reset the flag to allow the new process to start
                self.processing_running = False
                # Optional: Clear the progress bar visually to indicate a reset
                self.progress_var.set(0)
                self.status_var.set("Restarting...")
            else:
                # User clicked No, so we exit
                return

        if not self.input_path.get()  or not self.id_var.get():
            messagebox.showerror("Error", "Input path and ID are required")
            return

        # Collect parameters
        params : AppParameters = {
            "Input Path": self.input_path.get(),
            "STL Output Path": self.stl_output_path.get(),
            "Labelmap Output Path": self.labelmap_output_path.get(),
            "Scan Indicators": self.selected_indicators, #type: ignore
            "Group Filter": self.group_filter_var.get().strip(),
            "Use Default Indicators": self.use_default_indicators.get(),
            "ID": self.id_var.get(),
            "Configuration": self.config_var.get(),
            "Folds": [i for var, i in self.folds_checkbuttons if var.get()],
            'Split' : self.split_nifti_var.get(),
            'cut_enabled': self.enable_cut.get(),
            'lower_x': self.lower_x.get(),
            'upper_x': self.upper_x.get(),
            'lower_y': self.lower_y.get(),
            'upper_y': self.upper_y.get(),
            'lower_z': self.lower_z.get(),
            'upper_z': self.upper_z.get(),
            'keep_originals': self.keep_originals.get(),
            'use_percent': self.use_percent.get(), 
            "used_lps": self.used_lps.get(),
            'use_meshrepair': self.meshfix_var.get(),
            'remove_islands' : self.islands_var.get(), 
            'name_only' : self.just_name.get(),
            "nifti_input" : self.input_nifti.get(), 
            "run_analytics": self.analytics_var.get()
        }

        # Show summary of parameters
        result = "\n".join(f"{k}: {v}" for k, v in params.items())
        confirmation = messagebox.askyesno(
            "Confirm Parameters", 
            f"The following parameters will be used:\n\n{result}\n\nContinue with processing?"
        )
        
        if confirmation:
            self.processing_running = True
            self.status_var.set("Processing started...")
            self.progress_var.set(0)
            
            # Use a separate thread to avoid freezing the GUI
            # For simplicity, we're not using threading here, but in a real app you might want to
            processing_thread = threading.Thread(target=self.process_data, args=(params,))
            processing_thread.daemon = True
            processing_thread.start()


    def edit_segment_params(self):

        selected_id = self.id_var.get()
        if not selected_id:
            messagebox.showerror("Error", "Please select an ID or name first.")
            return
        if self.processing_running:
            messagebox.showwarning("Processing in Progress", 
                                  "Please wait for the current operation to complete before editing parameters.")
        segment_window = tk.Toplevel(self.root)
        segment_window.title("Edit Segment Params")
        segment_window.geometry("500x700")
        
        # Create a frame with scrollbar for the parameters
        scrolled = ScrolledFrame(segment_window, autohide=True)
        scrolled.pack(fill="both", expand=True, padx=5, pady=5)

        # Store references to Entry widgets
        param_entries = {}

        # Add parameter entries for the selected ID
        segment_frame = ttk.LabelFrame(scrolled, text=f"Segment Parameters for ID {selected_id}")
        segment_frame.pack(fill="x", expand=True, padx=10, pady=5)

        param_entries[selected_id] = {}

        row = 0
        for label_id, params in segment_params[selected_id].items():
            label_frame = ttk.LabelFrame(segment_frame, text=f"Label {label_id} ({params['label']})")
            label_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
            label_frame.columnconfigure(1, weight=1)  # Make entry column expand
            
            inner_row = 0
            for param_name, param_value in params.items():
                if param_name != 'label' and isinstance(param_value, (int, float, str)):
                    ttk.Label(label_frame, text=f"{param_name}:").grid(
                        row=inner_row, column=0, sticky="w", padx=5, pady=2
                    )
                    entry = ttk.Entry(label_frame)
                    entry.insert(0, str(param_value))
                    entry.grid(row=inner_row, column=1, sticky="ew", padx=5, pady=2)
                    param_entries[selected_id][(label_id, param_name)] = entry
                    inner_row += 1
            
            row += 1

        # Configure column weights for proper expansion
        segment_frame.columnconfigure(0, weight=1)

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
        button_frame = ttk.Frame(scrolled)
        button_frame.pack(fill="x", expand=True, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save", command=save_params, style="success").pack(side="left", padx=15)
        ttk.Button(button_frame, text="Cancel", command=segment_window.destroy, style="warning").pack(side="right", padx=15)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    from multiprocessing import freeze_support
    freeze_support()
    app = ParameterGUI(root)
    root.mainloop()
