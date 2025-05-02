import torch
import numpy as np
from scipy.ndimage import label as nd_label
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, Spacingd, ToNumpyd, SqueezeDimd
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from tqdm import tqdm
from monai.data.meta_tensor import MetaTensor

def plot_segmentation_results(inputs, outputs, labels, slice_idx=None, titles=None, cmap_inputs='gray', cmap_outputs='hot'):
    """
    General function to plot input images, model outputs, and ground truth labels side by side.

    Args:
        inputs (Tensor): Input MRI images (B, C, H, W, D).
        outputs (Tensor): Model output predictions (B, C, H, W, D).
        labels (Tensor): Ground truth segmentation masks (B, C, H, W, D).
        slice_idx (int, optional): The index of the slice along the depth axis to visualize. Defaults to middle slice.
        titles (list of str, optional): Titles for the subplots.
        cmap_inputs (str): Colormap for input images.
        cmap_outputs (str): Colormap for outputs and labels.
    """
    if slice_idx is None:
        slice_idx = inputs.shape[4] // 2  # Middle slice if not specified

    if titles is None:
        titles = ["Input Image", "Model Output (Sigmoid)", "Label"]

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(titles[0])
    plt.imshow(inputs[0, 0, :, :, slice_idx].cpu(), cmap=cmap_inputs)

    plt.subplot(1, 3, 2)
    plt.title(titles[1])
    plt.imshow(outputs[0, 0, :, :, slice_idx].cpu(), cmap=cmap_outputs)

    plt.subplot(1, 3, 3)
    plt.title(titles[2])
    plt.imshow(labels[0, 0, :, :, slice_idx].cpu(), cmap=cmap_inputs)

    plt.tight_layout()
    plt.show()

    plt.imshow(inputs[0, 0, :, :, slice_idx].cpu(), cmap="gray")
    plt.imshow(outputs[0, 0, :, :, slice_idx].cpu(), cmap="hot", alpha=0.5)
    plt.imshow(labels[0, 0, :, :, slice_idx].cpu(), cmap="Blues", alpha=0.3)


def analyze_validation_outputs(val_loader, model, device, threshold=0.5, num_batches=3):
    """
    Analyzes the outputs of the trained segmentation model on validation data.
    
    Args:
        val_loader (DataLoader): PyTorch DataLoader for the validation dataset.
        model (nn.Module): Trained PyTorch model for tumor segmentation.
        device (torch.device): Device to run the inference on (e.g., 'cuda' or 'cpu').
        threshold (float): Threshold for segmentation prediction (unused in this visualization).
        num_batches (int): Number of validation batches to visualize.
    """
    model.eval()
    outputs_hist = []
    
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            if idx >= num_batches:
                break

            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            val_outputs = torch.sigmoid(model(val_inputs))

            # Collect histogram data for output distribution
            outputs_hist.append(val_outputs.cpu().numpy().flatten())

            # Use the general plotting function
            plot_segmentation_results(val_inputs, val_outputs, val_labels)

    # Flatten all collected outputs and plot histogram of the sigmoid output values
    outputs_hist = np.concatenate(outputs_hist)
    plt.figure(figsize=(8, 5))
    plt.hist(outputs_hist, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Model Outputs (Sigmoid Values)")
    plt.xlabel("Output Value")
    plt.ylabel("Frequency")
    plt.show()


def filter_blobs(pred, min_size=50, max_size=5000):
    """
    Removes small or excessively large connected components (blobs) from a binary prediction mask.
    Args:
        pred (Tensor): Binary prediction mask.
        min_size (int): Minimum blob size to keep.
        max_size (int): Maximum blob size to keep.
    Returns:
        Tensor: Filtered binary mask.
    """
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    labeled_array, num_features = nd_label(pred_np)
    filtered = np.zeros_like(pred_np)

    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        size = np.sum(component)
        if min_size <= size <= max_size:
            filtered[component] = 1

    return torch.from_numpy(filtered).to(pred.device).type_as(pred)

def merge_labels(label):
    """
    Merges tumor-related labels into a single class.
    Converts label '2' to '1' while keeping other labels unchanged.
    Args:
        label (Tensor): Ground truth label mask.
    Returns:
        Tensor: Modified label mask.
    """
    return torch.where(label == 2, torch.tensor(1.0, dtype=label.dtype, device=label.device), label)

def plot_training_curves(history):
    """
    Plots training loss and validation Dice score curves.
    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_dice' lists.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(history['val_dice']) * 5, 5), history['val_dice'], label="Validation Dice", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

def adjust_learning_rate(optimizer, epoch, base_lr, warmup_epochs, total_epochs):
    """
    Adjusts the learning rate with linear warmup and cosine annealing.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        base_lr (float): The maximum learning rate.
        warmup_epochs (int): Number of warmup epochs.
        total_epochs (int): Total number of epochs.
    """
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = 0.5 * base_lr * (1 + torch.cos(torch.tensor(progress * torch.pi)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr.item() if isinstance(lr, torch.Tensor) else lr

def dice_debug(y_pred, y_true, epsilon=1e-8):
    """
    Computes Dice score with debug info for binary predictions.
    Args:
        y_pred (Tensor): Predicted mask, binary (0/1), shape (B, 1, H, W, D)
        y_true (Tensor): Ground truth mask, binary (0/1), same shape
        epsilon (float): Small value to avoid division by zero
    """
    assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
    y_pred_bin = (y_pred > 0.5).float()
    y_true_bin = (y_true > 0.5).float()

    intersection = (y_pred_bin * y_true_bin).sum(dim=(1, 2, 3, 4))
    pred_volume = y_pred_bin.sum(dim=(1, 2, 3, 4))
    true_volume = y_true_bin.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + epsilon) / (pred_volume + true_volume + epsilon)

    for i in range(y_pred.shape[0]):
        print(f"🧪 Sample {i+1}:")
        print(f"   ➤ Prediction voxels: {int(pred_volume[i].item())}")
        print(f"   ➤ Ground Truth voxels: {int(true_volume[i].item())}")
        print(f"   ➤ Intersection voxels: {int(intersection[i].item())}")
        print(f"   🎯 Dice: {dice[i].item():.4f}")
        print("-" * 40)

    return dice


def compute_dice_manual(preds, targets, skip_empty=True, eps=1e-8):
    """
    Compute Dice score manually for a batch of predictions and targets.

    Args:
        preds (Tensor): Binary prediction mask (B, 1, H, W, D).
        targets (Tensor): Ground truth binary mask (B, 1, H, W, D).
        skip_empty (bool): If True, ignore samples with no foreground in target.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Average Dice score across batch (excluding empty targets if skip_empty=True).
    """
    dices = []

    for i in range(preds.shape[0]):
        p = preds[i]
        t = targets[i]
        intersection = torch.sum(p * t).float()
        union = torch.sum(p).float() + torch.sum(t).float()

        if skip_empty and torch.sum(t) == 0:
            continue  # skip empty target

        dice = (2.0 * intersection + eps) / (union + eps)
        dices.append(dice.item())

    if len(dices) == 0:
        return 0.0
    return sum(dices) / len(dices)
    

def save_volume_nifti(tensor, meta_dict, path):
    """Save a tensor with MONAI metadata to NIfTI using the affine from metadata if present."""
    data_np = tensor.detach().cpu().numpy()

    # Safely get affine from metadata
    affine = None
    if isinstance(meta_dict, dict):
        if "original_affine" in meta_dict:
            affine = meta_dict["original_affine"]
        elif "affine" in meta_dict:
            affine = meta_dict["affine"]

    if isinstance(affine, torch.Tensor):
        affine = affine.cpu().numpy()
    if affine is None or affine.shape != (4, 4):
        affine = np.eye(4)

    nib.save(nib.Nifti1Image(data_np, affine), path)


def save_ranked_predictions_for_slicer(model, test_loader, save_dir="slicer_exports", device="cuda", N=3):
    """
    Run inference on the test set, compute Dice per sample, and save N best, worst, and mid samples for 3D Slicer.

    Args:
        model: Trained segmentation model.
        test_loader: DataLoader for test set.
        save_dir (str): Directory to save NIfTI files.
        device (str): Torch device.
        N (int): Number of best/mid/worst examples to save.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    scores = []
    batches = []
    dice_metric = DiceMetric(include_background=False, reduction="none")

    print("Evaluating batches...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating batches"):
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)
            outputs = torch.sigmoid(sliding_window_inference(inputs, roi_size=(128, 128, 128),
                                                             sw_batch_size=1, predictor=model, overlap=0.25))
            preds = (outputs > 0.5).float()
            batch_dice = dice_metric(preds, targets).cpu().numpy()

            for i in range(preds.shape[0]):
                scores.append(batch_dice[i])
                batches.append({
                    "image": batch["image"][i],
                    "label": batch["label"][i],
                    "pred": preds[i, 0].detach().cpu().numpy(),
                    "image_meta_dict": batch["image_meta_dict"],
                    "label_meta_dict": batch["label_meta_dict"]
                })

    # Sort by Dice and ensure unique selections
    scores = np.array(scores)
    sorted_idx = np.argsort(scores)
    total = len(scores)

    if total < 3 * N:
        print(f"⚠️ Not enough samples ({total}) to save {3 * N} distinct cases. Reducing N.")
        N = total // 3

    worst_idx = sorted_idx[:N]
    mid_start = max(0, total // 2 - N // 2)
    mid_idx = sorted_idx[mid_start:mid_start + N]
    best_idx = sorted_idx[-N:]

    selected = []
    selected += [("worst", int(i), scores[int(i)]) for i in worst_idx]
    selected += [("mid", int(i), scores[int(i)]) for i in mid_idx]
    selected += [("best", int(i), scores[int(i)]) for i in best_idx]

    for count, (label, idx, _) in enumerate(selected):
        batch = batches[idx]
        pred_np = batch["pred"]

        image_tensor = batch["image"][0]
        label_tensor = batch["label"][0]
        image_meta = batch["image_meta_dict"]
        label_meta = batch["label_meta_dict"]

        save_volume_nifti(image_tensor, image_meta, os.path.join(save_dir, f"case_{count}_{label}_image.nii.gz"))
        save_volume_nifti(label_tensor, label_meta, os.path.join(save_dir, f"case_{count}_{label}_label.nii.gz"))
        save_volume_nifti(torch.from_numpy(pred_np), image_meta, os.path.join(save_dir, f"case_{count}_{label}_prediction.nii.gz"))

    print(f"✅ Saved {len(selected)} volumes (best, worst, mid) to {save_dir}")
