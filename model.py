from monai.networks.nets import UNet, DynUNet, AttentionUnet
from monai.networks.layers import Norm
import torch
from config import device

def get_model(model_name="dynunet", deep_supervision=True):
    """
    Returns the requested model architecture.

    Args:
        model_name (str): Model type. Options: "unet", "dynunet", "dynunet_no_ds", "attention_unet".
        deep_supervision (bool): Only relevant for DynUNet.

    Returns:
        nn.Module: Configured model.
    """
    if model_name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE
        )
    elif model_name == "dynunet":
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=[16, 32, 64, 128, 256],
            norm_name=Norm.INSTANCE,
            deep_supervision=deep_supervision
        )
    elif model_name == "dynunet_no_ds":
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=[16, 32, 64, 128, 256],
            norm_name=Norm.INSTANCE,
            deep_supervision=False
        )
    elif model_name == "attention_unet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available options: 'unet', 'dynunet', 'dynunet_no_ds', 'attention_unet'.")

    return model.to(device)
