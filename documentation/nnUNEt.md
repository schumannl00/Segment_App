# nnUNet

## How to integrate new models.

- modify the ids.json by adding the corresponding new block
  ```
  "222": {
        "body_part": "Arm",
        "Path_to_results": {
            "3d_fullres": "Dataset222_Arm/nnUNetTrainer__nnUNetPlans__3d_fullres"   #modify so it fits the tariner and config used
        },
        "configurations": [
            "3d_fullres"
        ],
        "prefix": "Arm_", #always 3 letters with _
        "suffix": "_0000"  # stays the same, this is only relevant for MRI scans where we might have a T1 and T2
         # I do not think doing the same scan in a bone and soft tissue window would improve much and would just cause data leakage. 
    }  
    ```

- modify the labels.json: 
  ```
   "222": {
        "1": "Humerus",
        "2": "Radius",
        "3": "Ulna"
    },
  ```


## How to actually train a new model
- follow the nnUNet guideline: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/README.md for the commands
- Handbook: 
    - have your raw data (Dicoms or NIFTI) with the corresponding masks or stls (you can start with 20 train 1 fold and use that to segment the rest and make some small fixes, having 50 or so scans in the end is fine for easy structures go to 200 or so for harder stuff) 
    - variance is more important than amount so include some messy, ugly scans
    - nifti + nifti mask would be ideal but we will go with dicoms + stl from mimics
    - have your raw files ready and already integrate the model into the app, then run the conversion task from dicom to nifti and kill the app (will make a docker image for the whole process, combines the conversion with the renaming will add name later here)
    - check decoder.json, sometimes stuff gets mixed around
    - this gives you nnUNet style name for the files
    - use 3D slicer to convert the stls into label maps
    - **The order has to be consistent as in 3DSlicer**: so the first segment gets map to 1, the second to 2, do not mix them up. Best to make a physical note and be mindful when doing it. This will cause problems that are super hard to track if that gets mesed up. This order is the one used in the final dataset.json for training. 
    -  safe and rename in the nnUnet format (this is a nice sanity check, so if the numbers do not match, something went wrong)
    -  if data is done move to nnUNet_raw  folder and follow nnUnet preprocessing guide i.e. run 
    ``` nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity ```
    (This is the all in one solution if you do not plan to change anything in the plans. )
    - use the train_all_folds.ps1 script to start training, make adjustments as needed (so different trainer like the 250 epoch one or so) see the trainer variants folder in nnUNet to see what they have stock in addition to our custom trainers 
  
**Video/ Screengrab needed ?**

## Advice for custom plans


 - Consider respacing the data, if we need global context this is where we kinda dial in global context, it also makes inference faster. A smaller spacing makes teh results smoother as we have more data and do not need to interpolate but it starts loosing the global view, as with 0.5mm spacing and the standard 128^3 patches we only see a 6 cm cube at a time. This can cause problemns with similiar structures or big artifacts, 1mm seems to be fine. 
  - Also expermient with aniso patch sizes like 160x128x128  (z,y,x)
  - use respacing function from /misc (might containerize for easier use)
  - you can also use linear interpolation in the lowres setting for the image, is likely fine and makes training faster, will look blockier but no overshoot artifacts 

## Custom trainers 

We have 3 custom trainers so far: 
1. CustomStrongerAug: can be used as a base trainer, used it for fine tuning but base training is fine as well. Change 
```
self.head_weight = 1  #bump up for finetuning to 5 or so, especially if the body is fine 
self.finetune_run = False # set to True for finetuning 
```
2. nnUNetTrainerNewHead : for transfer learning, so if we add a new head with more classes it copies over the old stuff, perfect for finetuning on a area which was used before, but needs a new part (used it for ankle to add calcaneus)
   
3. UniversalAdapter (LORA): The big shift, we add LORA blocks in a conv adapter style (so a down and upsample conv instead of the usual linear block), added SiLu as a non-linearity and quite high dropout. usual modern LR setup with Warmup and CosineAnnealing. This is a fickle thing, so do not touch it. It fights with  the way nnUNet wants to be used, so i needed to overwrite a lot. It shows promise in hard finetuning cases where we do not have a lot of training data like pediatric scoliosis as it prevenmts overfitting and can adapt to e.g. geometric changes. The pretained weights are hard coded here, I could not get it to work with the pretrained weight flag the way I wanted to. 

**Important:** They all have mlflow logging which I would advise in using as well, just run 
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 
```

If you keep it running locally no need for auth tbh, otherwise run 

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 --app-name basic-auth --gunicorn-opts "--timeout 120"
``` 
This should be done in WSL with all the dependencies installed (needs flask and so one), so only do if needed. Nice to do with a Cloudflare Tunnel so it is accessible without VPN from any device with 

```
cloudflared tunnel --url http://localhost:5000 
```


If this becomes more commonly used, maybe make a VM for the server and route accordingly with a stronger DB like postgres (run with a simple container)