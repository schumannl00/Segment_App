import SimpleITK as sitk

#we had problems with the headers when creating masks from stl models with 3DSlicer, this fixes that, modify to your need  


for i in range(18,79):  

    number = str(i).zfill(3)
    print(number)
    image_path = rf"D:\nnUNet_raw\Dataset121_Ganzbein\imagesTr\Leg_{number}_0000.nii.gz"
    segmentation_path = rf"D:\nnUNet_raw\Dataset121_Ganzbein\labelsTr\Leg_{number}.nii.gz"
    try: 
        image_sitk = sitk.ReadImage(image_path)
        segmentation_sitk = sitk.ReadImage(segmentation_path)

        coregisterer = sitk.ResampleImageFilter()
        coregisterer.SetReferenceImage(image_sitk)
        coregisterer.SetInterpolator(sitk.sitkNearestNeighbor)

        coregistered_segmentation_sitk = coregisterer .Execute(segmentation_sitk)

        sitk.WriteImage(coregistered_segmentation_sitk, segmentation_path)
    except: 
        print(f"file with {image_path} does not exist")