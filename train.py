import torch
import time
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import torch.nn.functional as F

from config import (
    device as default_device, model_path as default_model_path,
    validate_every as default_validate_every, fast_until_epoch as default_fast_until,
    fast_roi_size as default_fast_roi_size, full_roi_size as default_full_roi_size,
    fast_overlap as default_fast_overlap, full_overlap as default_full_overlap,
    fast_sw_batch_size as default_fast_batch, full_sw_batch_size as default_full_batch
)

from utils import filter_blobs, plot_training_curves, adjust_learning_rate
from metrics import compute_dice

def train_and_validate(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    max_epochs=300,
    device=None,
    model_path=default_model_path,
    save_name="best_model.pth",
    load_existing=False,
    validate_every=default_validate_every,
    fast_until_epoch=default_fast_until,
    fast_roi_size=default_fast_roi_size,
    full_roi_size=default_full_roi_size,
    fast_overlap=default_fast_overlap,
    full_overlap=default_full_overlap,
    fast_sw_batch_size=default_fast_batch,
    full_sw_batch_size=default_full_batch,
    debug=False,
    early_training=False,
    early_stop_patience=10,
    early_stop_min_delta=0.001,
    early_stop_min_num_epochs=50
):
    device = device or default_device
    scaler = GradScaler()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    save_path = model_path + save_name

    if load_existing:
        print(f"🔄 Loading model weights from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    model.to(device)
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': []}
    start_time = time.time()

    best_soft_dice = 0.0
    no_improvement_counter = 0

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0

        adjust_learning_rate(optimizer, epoch, base_lr=1e-4, warmup_epochs=10, total_epochs=max_epochs)
        if debug:
            print(f"   [DEBUG] Current Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = 0

                if isinstance(outputs, (list, tuple)):  # Deep supervision
                    for o in outputs:
                        if o.shape[1:] != labels.shape[1:]:
                            o = F.interpolate(o, size=labels.shape[2:], mode="trilinear", align_corners=False)
                        if o.shape[0] != labels.shape[0]:
                            min_batch = min(o.shape[0], labels.shape[0])
                            o = o[:min_batch]
                            labels = labels[:min_batch]
                        loss += loss_fn(o, labels)
                    loss /= len(outputs)
                else:
                    if outputs.shape[1:] != labels.shape[1:]:
                        outputs = F.interpolate(outputs, size=labels.shape[2:], mode="trilinear", align_corners=False)
                    if outputs.shape[0] != labels.shape[0]:
                        min_batch = min(outputs.shape[0], labels.shape[0])
                        outputs = outputs[:min_batch]
                        labels = labels[:min_batch]
                    loss = loss_fn(outputs, labels)

                if debug:
                    print(f"   [DEBUG] Loss: {loss.item():.6f}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        print(f"   🔹 Avg Train Loss: {avg_loss:.4f}")

        # ===== Validation =====
        do_validate = (epoch % validate_every == 0) or (epoch >= fast_until_epoch)
        if do_validate:
            model.eval()
            use_fast = early_training or epoch < fast_until_epoch
            roi_size = fast_roi_size if use_fast else full_roi_size
            overlap = fast_overlap if use_fast else full_overlap
            sw_batch_size = fast_sw_batch_size if use_fast else full_sw_batch_size
        
            soft_dice_epoch = 0.0
            num_val_batches = 0
        
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size=roi_size, sw_batch_size=sw_batch_size,
                        predictor=model, overlap=overlap
                    )
        
                    sigmoid_outputs = torch.sigmoid(val_outputs)
                    soft_dice = compute_dice(sigmoid_outputs, val_labels)
        
                    soft_dice_epoch += soft_dice
                    num_val_batches += 1
        
                    if debug:
                        print(f"   [DEBUG] Soft Dice (batch): {soft_dice:.6f}")
        
            avg_soft_dice = soft_dice_epoch / num_val_batches
            history['val_dice'].append(avg_soft_dice)
            print(f"   🎯 Validation Soft Dice: {avg_soft_dice:.4f}")
        
            # ===== Early Stopping Logic =====
            if avg_soft_dice > best_soft_dice + early_stop_min_delta:
                best_soft_dice = avg_soft_dice
                no_improvement_counter = 0
            else:
                if epoch >= early_stop_min_num_epochs:
                    no_improvement_counter += 1
                    print(f"   ⚠️ No soft Dice improvement. Counter: {no_improvement_counter}/{early_stop_patience}")
        
            if avg_soft_dice > best_dice:
                best_dice = avg_soft_dice
                torch.save(model.state_dict(), save_path)
                print(f"   💾 Best model saved at {save_path}.")
        
            if no_improvement_counter >= early_stop_patience:
                print(f"\n🛑 Early stopping triggered! No soft Dice improvement for {early_stop_patience} validations.")
                break

    total_minutes = (time.time() - start_time) / 60
    print(f"\n🕒 Total training time: {total_minutes:.2f} minutes")
    print(f"🏁 Best Validation Dice (thresholded): {best_dice:.4f}")
    plot_training_curves(history)

    return history, best_dice


