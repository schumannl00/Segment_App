import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from ttkbootstrap.scrolled import ScrolledFrame
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
import nibabel as nib
from nibabel import load, Nifti1Image, save 
from skimage import measure
from scipy import ndimage
import pyvista as pv
from stl import mesh
from multi_stl import process_directory_parallel
import shutil
from cutting import masking, zcut
from multiprocessed import raw_data_to_nifti_parallel, nifti_renamer
from modifier import merger, stl_renamer_with_lut
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import logging
from utils.logging_tool import gui_log_output, SuppressStdout, TerminalOnlyStdout

with open("ids.json", "r") as ids:
    id_dict = json.load(ids)
with open("labels.json", "r") as labels:
    labels_dict = json.load(labels)

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
        self.root.geometry("1100x650")
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
        path_frame.grid(row=0, column=0, columnspan=6, sticky = EW, pady=(0,10))
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
        indicator_frame.grid(row=1, column=0, columnspan=6, sticky=EW, pady=(0,10))
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

        self.indicators_menu = ttk.Menubutton(indicator_controls_frame, text="Select Indicators", width=60)
        self.indicators_menu.grid(row=0, column=0, sticky=EW, padx=(0, 5))
       
        self.dropdown_menu = tk.Menu(self.indicators_menu, tearoff=0)
        self.indicators_menu["menu"] = self.dropdown_menu

        # Custom Indicator Button
        self.add_custom_button = tb.Button(
            indicator_controls_frame, 
            text="+", 
            width=3, 
            command=self.add_custom_indicator, bootstyle= "info")
        
        self.add_custom_button.grid(row=0, column=1, padx=(5, 5))

        tb.Label(indicator_controls_frame, text="Group:").grid(row=0, column=2, padx=(10, 2), sticky=E)
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
            command=self.toggle_indicators_entry, bootstyle= "danger"
        )
        self.default_indicators_check.grid(row=0, column=4, padx=(10, 10))

       
        

        # Row 4: Select ID and Configurations

        config_group_frame = tb.LabelFrame(main_frame, text="Details for the segmentation", padding = "10", bootstyle= INFO)
        config_group_frame.grid(row=2, column=0, columnspan=1, sticky=EW, pady=(0, 10)) # Use grid in main_frame
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
            var = tk.BooleanVar(value=False)  
            check = tb.Checkbutton(self.folds_frame, text=f"Fold {i}", variable=var)
            check.grid(row=0, column=i, padx=5, pady=5)
            self.folds_checkbuttons.append((var, i))

        self.select_all_button = ttk.Button(self.folds_frame, text="Select All", command=self.select_all_folds)
        self.select_all_button.grid(row=1, column=0, columnspan=2, padx=15, pady=5)
        self.deselect_all_button = ttk.Button(self.folds_frame, text= "Deselect all", command=self.deselect_all_folds, style= WARNING )
        self.deselect_all_button.grid(row=1, column=2, columnspan=2, padx=15, pady=5 )
      
       #Processing
    
        preprocessing_frame = tb.LabelFrame(main_frame, text="Processing Options", padding = "10", bootstyle= INFO)
        preprocessing_frame.grid(row=2, column=1, columnspan=4, sticky=EW, pady=(0, 5), padx=5) # Use grid in main_frame
        preprocessing_frame.columnconfigure(1, weight=1)




        self.split_nifti_var = tk.BooleanVar(value=False)
        tb.Label(preprocessing_frame, text= "Cutting along x direction:").grid(row=0, column =0, sticky = W, pady = 5, padx = 5)
        self.split_nifti_check = tb.Checkbutton(preprocessing_frame, text="Split NIFTIs into left and right", variable=self.split_nifti_var)
        self.split_nifti_check.grid(row=0, column=1, columnspan=4,   sticky=tk.W, pady=5, padx= 5)

        self.enable_zcut = tk.BooleanVar(value=False)
        enable_checkbox = tb.Checkbutton(preprocessing_frame, 
                                         text="Cut along z direction",
                                         variable=self.enable_zcut,
                                         command=self.toggle_zcut_inputs)
        enable_checkbox.grid(row=1, column=0, sticky=W, pady=5, padx = 5)
        
     
        self.keep_originals= tk.BooleanVar(value=False)
        self.keep_originals_checkbox = tb.Checkbutton(preprocessing_frame, text="Keep originals", variable=self.keep_originals, state=DISABLED)
        self.keep_originals_checkbox.grid(row=1, column=1, sticky=W, pady=5, padx=5)
       
        ttk.Label(preprocessing_frame, text="Lower:").grid(row=1, column=2, sticky=E, pady=(0, 10), padx=(0, 5))
        self.lower_var = tk.StringVar(value="0")
        self.lower_entry = tb.Entry(preprocessing_frame, textvariable=self.lower_var, width=10)
        self.lower_entry.grid(row=1, column=3, sticky=tk.W, pady=(0, 10))
        self.lower_entry["state"] = "disabled"
        
  
        ttk.Label(preprocessing_frame, text="Upper:").grid(row=1, column=4, sticky=tk.E, pady=(0, 10), padx=(20, 5))
        self.upper_var = tk.StringVar(value="0")
        self.upper_entry = tb.Entry(preprocessing_frame, textvariable=self.upper_var, width=10)
        self.upper_entry.grid(row=1, column=5, sticky=tk.W, pady=(0, 10))
        self.upper_entry["state"] = "disabled"

   
        self.multiple_ids_var = tk.BooleanVar(value=False)
        self.multiple_ids_check =tb.Checkbutton(preprocessing_frame, text="Does the Input Folder have indiviudual subfolders \n e.g. multiple patients? Use custom filters ('+' -Button) as  naming conventions might vary.", variable=self.multiple_ids_var)
        self.multiple_ids_check.grid(row=2, column=0, columnspan=6, sticky=W, pady=(0, 10))

 
        self.meshfix_var = tk.BooleanVar(value=True)
        self.meshfix_check = tb.Checkbutton(preprocessing_frame, text= "Appply Pymesh meshrepair (caps open stls as well)",  variable=self.meshfix_var, command=self.on_meshfix_toggle)
        self.meshfix_check.grid(row=3, column=0, columnspan=6, sticky = W, pady=(0,10))

    
        self.islands_var = tk.BooleanVar(value=True)
        self.islands_check = tb.Checkbutton(preprocessing_frame, text= "Remove all but the largest element from stl",  variable=self.islands_var)
        self.islands_check.grid(row=4, column=0, columnspan=6, sticky = W, pady=(0,10))


        # Button frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        submit_button = tb.Button(button_frame, text="Submit", command=self.submit, bootstyle= SUCCESS)
        submit_button.pack(side=tk.LEFT, padx=(0, 10))
        clear_button = tb.Button(button_frame, text="Clear", command=self.clear_fields, bootstyle = WARNING)
        clear_button.pack(side=tk.LEFT)

        # Progress bar for displaying operation progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tb.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
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
    def process_data(self, params):
        """Process the input data using the parameters from the GUI"""
        try:
            self.progress_queue.put(ProgressEvent(10, "Converting DICOM to NIfTI..."))
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
            multiple = params['Multiple']
            lower = params['z-lower']
            upper = params['z-upper']
            cut_z = params['cut_z']
            keep_originals= params["keep_originals"]
            meshrepair = params["use_meshrepair"]
            remove_islands = params["remove_islands"]
            # Ensure output directories exist
            os.makedirs(stl_output_path, exist_ok=True)
            os.makedirs(labelmap_output_path, exist_ok=True)
            #Step 0: Convert input directory into single folder if multiple selected
            if multiple: 
                self.progress_queue.put(ProgressEvent(5, "Merging multiple folders..."))
                input_path= merger(input_path)
            # Step 1: Convert DICOM to NIfTI 
            self.progress_queue.put(ProgressEvent(15, "Converting DICOM to NIfTI..."))
            
            raw_data_to_nifti_parallel(input_path, scans_indicators=scan_indicators, group_filter=group_filter, use_default=use_default, max_workers=12)
            # Step 2: Process NIfTI files - get files to process
            self.progress_queue.put(ProgressEvent(30, "Processing NIfTI files..."))
            
            inference_path = os.path.join(str(input_path.parent),r'NIFTI')
           
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
                    
                    for folder in os.listdir(inference_path):
                        zcut(os.path.join(inference_path,folder),lower,upper, keep_original=keep_originals, backup_dir=input_path.parent/"uncut_nii")
                
                except Exception as e:
                    messagebox.showerror("The NIFTIs cannot be cut along the z-axis, str({e}).")
                    return
           
            
                
                    
           
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
            process_directory_parallel(labelmap_output_path, stl_output_path, segment_params=segment_params[selected_id], split=split, use_pymeshfix=meshrepair, remove_islands=remove_islands)
            
            self.progress_queue.put(ProgressEvent(90, "STL names back to original names..."))
            stl_renamer_with_lut(stl_output_path=stl_output_path, file_mapping=file_mapping)
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
            messagebox.showwarning("Processing in Progress", 
                                  "Please wait for the current operation to complete.")
            return

        if not self.input_path.get()  or not self.id_var.get():
            messagebox.showerror("Error", "Input path and ID are required")
            return

        # Collect parameters
        params = {
            "Input Path": self.input_path.get(),
            "STL Output Path": self.stl_output_path.get(),
            "Labelmap Output Path": self.labelmap_output_path.get(),
            "Scan Indicators": self.selected_indicators,
            "Group Filter": self.group_filter_var.get().strip(),
            "Use Default Indicators": self.use_default_indicators.get(),
            "ID": self.id_var.get(),
            "Configuration": self.config_var.get(),
            "Folds": [i for var, i in self.folds_checkbuttons if var.get()],
            'Split' : self.split_nifti_var.get(),
            'Multiple': self.multiple_ids_var.get(),
            'z-lower': self.lower_var.get(),
            'z-upper': self.upper_var.get(),
            'cut_z': self.enable_zcut.get(),
            'keep_originals' : self.keep_originals.get(),
            'use_meshrepair': self.meshfix_var.get(),
            'remove_islands' : self.islands_var.get()
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
