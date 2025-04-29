# if using google colab pro+, you would need to reinstall these libraries.
!pip install pyvista 
!pip install medpy 

import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import torch.nn.functional as F
from medpy.metric import binary
from skimage import measure
from scipy import ndimage
import pyvista as pv
import pandas as pd
from monai.networks.nets import UNet

case_ids = [
    "BraTS20_Training_337",
]


val_root = "/content/drive/MyDrive/4thyearproject_code/TEST"
model_path = "/content/drive/MyDrive/4thyearproject_code/pretrained_models/BESTMODEL_unet_se_encoder_final.pt"
output_root = "/content/drive/MyDrive/4thyearproject_code/EVALUATION_OUTPUT17"
os.makedirs(output_root, exist_ok=True)

# defining the model, this is similar to the previous codes.
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = SEBlock3D(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        return self.attn(self.conv(x))

class UNetSEEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=(16, 32, 64, 128, 256)):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels[0], use_se=True)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(base_channels[0], base_channels[1], use_se=True)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(base_channels[1], base_channels[2], use_se=True)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock(base_channels[2], base_channels[3], use_se=True)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base_channels[3], base_channels[4])
        self.up4 = nn.ConvTranspose3d(base_channels[4], base_channels[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels[4], base_channels[3])
        self.up3 = nn.ConvTranspose3d(base_channels[3], base_channels[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels[3], base_channels[2])
        self.up2 = nn.ConvTranspose3d(base_channels[2], base_channels[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels[2], base_channels[1])
        self.up1 = nn.ConvTranspose3d(base_channels[1], base_channels[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels[1], base_channels[0])

        self.final_conv = nn.Conv3d(base_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final_conv(d1)

model = UNetSEEncoder(in_channels=4, out_channels=4).to("cuda")
model.load_state_dict(torch.load(model_path, map_location="cuda"))
model.eval()

# padding to multiples of 16 to suit the UNet
def pad_to_multiple(tensor, multiple=16):
    shape = tensor.shape
    if len(shape) == 4:
        _, d, h, w = shape
        pad_dims = (0, (multiple - w % multiple) % multiple,
                    0, (multiple - h % multiple) % multiple,
                    0, (multiple - d % multiple) % multiple)
    elif len(shape) == 3:
        d, h, w = shape
        pad_dims = (0, (multiple - w % multiple) % multiple,
                    0, (multiple - h % multiple) % multiple,
                    0, (multiple - d % multiple) % multiple)
    else:
        raise ValueError(f"Unsupported tensor shape: {shape}")

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    return F.pad(tensor, pad_dims, mode='constant', value=0)

# This is essentially the beginning of the Marching Cubes algorithm where it extracts the surfaces then reconstructs them in triangulation.
# The Laplacian smoothing is also incorporated here with the smooth_iters function.
# Tis extracts the entire brain surface of the predicted segmentation file and takes into account the subregion segmentation colourations. 
# would have been better if this was transparent and the actual tumour is visible from the inside too, maybe for the viva
def extract_surface(segmentation, label, spacing=(1.0, 1.0, 1.0), smooth_iters=30):
    binary = (segmentation == label).astype(np.uint8) if label != 'brain' else segmentation
    if np.sum(binary) == 0:
        return None
    verts, faces, _, _ = measure.marching_cubes(binary, level=0.5, spacing=spacing)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    mesh = pv.PolyData(verts, faces)
    return mesh.smooth(n_iter=smooth_iters)

# volume is computed 
def compute_volumes(segmentation, spacing):
    voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000  
    return {
        "Vol_Whole": np.sum(np.isin(segmentation, [1, 2, 3])) * voxel_volume,
        "Vol_Core": np.sum(np.isin(segmentation, [1, 3])) * voxel_volume,
        "Vol_Enhancing": np.sum(segmentation == 3) * voxel_volume,
    }

# distance from tumour centroid to cranial margin and the midline is computed
def compute_distances(segmentation, spacing):
    binary_mask = (segmentation > 0).astype(np.uint8)
    coords = np.argwhere(binary_mask)
    if coords.shape[0] == 0:
        return {"Midline_Dist_mm": None, "Cranial_Dist_mm": None, "Cranial_Margin_Dist_mm": None}
    centroid_voxel = coords.mean(axis=0)
    _, _, width = segmentation.shape
    midline_x = width / 2
    midline_dist_mm = abs(centroid_voxel[2] - midline_x) * spacing[2]
    min_z = coords[:, 0].min()
    cranial_margin_dist_mm = abs(centroid_voxel[0] - min_z) * spacing[0]
    cranial_dist_mm = centroid_voxel[0] * spacing[0]
    return {
        "Midline_Dist_mm": midline_dist_mm,
        "Cranial_Dist_mm": cranial_dist_mm,
        "Cranial_Margin_Dist_mm": cranial_margin_dist_mm,
    }

# dice score per subregion is computed
def dice_score(pred, gt, region):
    if region == "WT":
        pred_bin = np.isin(pred, [1, 2, 3]).astype(np.uint8)
        gt_bin = np.isin(gt, [1, 2, 3]).astype(np.uint8)
    elif region == "TC":
        pred_bin = np.isin(pred, [1, 3]).astype(np.uint8)
        gt_bin = np.isin(gt, [1, 3]).astype(np.uint8)
    elif region == "ET":
        pred_bin = (pred == 3).astype(np.uint8)
        gt_bin = (gt == 3).astype(np.uint8)
    else:
        raise ValueError("unknown region")

    if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
        return 1.0
    elif np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
        return 0.0
    else:
        return binary.dc(pred_bin, gt_bin)

# 95% hausdorff distance per subregion is computed
def hausdorff(pred, gt, region):
    if region == "WT":
        pred_mask = np.isin(pred, [1, 2, 3])
        gt_mask = np.isin(gt, [1, 2, 3])
    elif region == "TC":
        pred_mask = np.isin(pred, [1, 3])
        gt_mask = np.isin(gt, [1, 3])
    elif region == "ET":
        pred_mask = (pred == 3)
        gt_mask = (gt == 3)
    else:
        raise ValueError("unknown region")

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return np.nan

    return binary.hd95(pred_mask.astype(np.uint8), gt_mask.astype(np.uint8))

# boundary dice per subregion is calculated. higher margins means there is more erosion of the voxels and it would be a less 
# precise match. this is because it would then increase the bounds of what is considered a boundary voxel and this might include misalignments. 
def boundary_dice(pred, target, class_idx, margin=2):
    pred_np = (pred == class_idx).astype(np.uint8)
    target_np = (target == class_idx).astype(np.uint8)

    if not np.any(pred_np) or not np.any(target_np):
        return np.nan

    structure = ndimage.generate_binary_structure(3, 1)
    pred_eroded = ndimage.binary_erosion(pred_np, structure=structure, iterations=margin)
    target_eroded = ndimage.binary_erosion(target_np, structure=structure, iterations=margin)

    pred_boundary = np.logical_xor(pred_np, pred_eroded)
    target_boundary = np.logical_xor(target_np, target_eroded)
    intersection = np.logical_and(pred_boundary, target_boundary).sum()

    union = pred_boundary.sum() + target_boundary.sum()
    return 2 * intersection / union if union > 0 else np.nan

# validation
summary = []

for case_id in case_ids:
    print(f" Processing {case_id}...")
    case_path = os.path.join(val_root, case_id)
    out_path = os.path.join(output_root, case_id)
    os.makedirs(out_path, exist_ok=True)

    imgs = [nib.load(os.path.join(case_path, f"{case_id}_{mod}.nii")).get_fdata() for mod in ["flair", "t1", "t1ce", "t2"]]
    img = np.stack(imgs).astype(np.float32)
    for c in range(4):
        img[c] /= img[c].max() if img[c].max() > 0 else 1
    img = pad_to_multiple(img).unsqueeze(0).to("cuda")

    with torch.no_grad():
        pred = torch.argmax(model(img), dim=1).cpu().squeeze().numpy()

# adding flair as a reference point for the images
    ref_nii = nib.load(os.path.join(case_path, f"{case_id}_flair.nii"))
    gt = nib.load(os.path.join(case_path, f"{case_id}_seg.nii")).get_fdata().astype(np.uint8)
    gt[gt == 4] = 3
    pred = pred[:gt.shape[0], :gt.shape[1], :gt.shape[2]]

    spacing = ref_nii.header.get_zooms()[:3]
    labels = {1: "WT", 2: "TC", 3: "ET"}
    color_map = {1: "blue", 2: "green", 3: "red"}
    volumes = compute_volumes(pred, spacing)
    pred_nifti = nib.Nifti1Image(pred.astype(np.uint8), ref_nii.affine)
    nib.save(pred_nifti, os.path.join(out_path, f"{case_id}_pred_seg.nii.gz"))


    metrics = {
        "Case": case_id,
        "Dice_Whole": dice_score(pred, gt, "WT"),
        "Dice_Core": dice_score(pred, gt, "TC"),
        "Dice_Enhancing": dice_score(pred, gt, "ET"),
        "Hausdorff_Whole": hausdorff(pred, gt, "WT"),
        "Hausdorff_Core": hausdorff(pred, gt, "TC"),
        "Hausdorff_Enhancing": hausdorff(pred, gt, "ET"),
        "Boundary_Dice_Whole": boundary_dice(pred, gt, 1),
        "Boundary_Dice_Core": boundary_dice(pred, gt, 2),
        "Boundary_Dice_Enhancing": boundary_dice(pred, gt, 3),
        **volumes,
        **compute_distances(pred, spacing)
    }

    summary.append(metrics)

# would be nice if this was interactive
# pyvista is used to plot the masks and also take screenshots of the model in different angles
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"

    brain_mask = (imgs[1] > imgs[1].mean() * 0.5).astype(np.uint8)
    brain_mask = ndimage.binary_fill_holes(ndimage.binary_closing(brain_mask, structure=np.ones((5,5,5)))).astype(np.uint8)
    brain_mesh = extract_surface(brain_mask, label='brain', spacing=spacing)
    if brain_mesh:
        plotter.add_mesh(brain_mesh, color="lightgray", opacity=0.2)

    for label, name in labels.items():
        mesh = extract_surface(pred, label, spacing)
        if mesh:
            mesh.save(os.path.join(out_path, f"{case_id}_{name.lower()}.stl"))
            plotter.add_mesh(mesh, color=color_map[label])

    for view_name, cam_pos in {
        "isometric": (1, 1, 1),
        "anterior": (0, 0, 1),
        "posterior": (0, 0, -1),
        "superior": (0, -1, 0),
        "right": (1, 0, 0),
        "left": (-1, 0, 0)
    }.items():
        plotter.camera_position = cam_pos
        plotter.camera.zoom(1.3)
        screenshot_path = os.path.join(out_path, f"{case_id}_view_{view_name}.png")
        plotter.screenshot(screenshot_path)

    plotter.close()

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_root, "tumourvalidation_full_metrics_summary.csv"), index=False)
print("done")
