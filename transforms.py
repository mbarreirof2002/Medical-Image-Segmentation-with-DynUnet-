from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    SpatialPadd, RandFlipd, RandRotate90d, ToTensord, RandBiasFieldd,
    RandGaussianNoised, RandAdjustContrastd, Rand3DElasticd, RandAffined,
    LambdaD, ToTensorD
)
from utils import merge_labels

def get_train_transforms(use_strong_augmentation=True):
    if use_strong_augmentation:
        return Compose([
            LoadImaged(keys=["image", "label"]),
            LambdaD(keys="label", func=merge_labels),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=(128, 128, 128),
                pos=1, neg=0, num_samples=4, image_key="image", image_threshold=0, allow_smaller=True
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            ToTensord(keys=["image", "label"]),
            RandBiasFieldd(keys=["image"], prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            Rand3DElasticd(
                keys=["image", "label"], prob=0.3, sigma_range=(5, 8), magnitude_range=(100, 200),
                rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"), padding_mode="border"
            ),
            RandAffined(
                keys=["image", "label"], prob=0.3, rotate_range=(0.1, 0.1, 0.1),
                shear_range=(0.05, 0.05, 0.05), scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest")
            )
        ])
    else:
        return Compose([
        LoadImaged(keys=["image", "label"]),
        LambdaD(keys="label", func=merge_labels),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=(128, 128, 128),
            pos=1, neg=0, num_samples=4, image_key="image", image_threshold=0, allow_smaller=True
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0), 
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=1),    
        ToTensord(keys=["image", "label"]),
        RandBiasFieldd(keys=["image"], prob=0.1),                      
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01), 
        RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.9, 1.1)),   
        RandAffined(                                                  
            keys=["image", "label"], prob=0.2, 
            rotate_range=(0.05, 0.05, 0.05), 
            shear_range=(0.02, 0.02, 0.02), 
            scale_range=(0.05, 0.05, 0.05), 
            mode=("bilinear", "nearest")
        )
    ])

        

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        LambdaD(keys="label", func=merge_labels),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=(128, 128, 128),
            pos=3, neg=1, num_samples=1, image_key="image", image_threshold=0, allow_smaller=True
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        ToTensorD(keys=["image", "label"])
    ])

def get_test_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        LambdaD(keys="label", func=merge_labels),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=(128, 128, 128),
            pos=3, neg=1, num_samples=1, image_key="image", image_threshold=0, allow_smaller=True
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["image", "label"])
    ])



