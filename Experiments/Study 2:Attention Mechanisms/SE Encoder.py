!pip install monai
!pip install torchinfo
!pip install numpy==1.26.4 --force-reinstall --no-cache-dir

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import csv
import pandas as pd
import tqdm
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torchinfo import summary


from metrics import compute_all_metrics

# Reproducibility
torch.manual.seed (42)
np.random.seed (42)
if torch.cuda.is_available():
   torch.cuda.manual_seed_all (42)

# Setup for training
BATCH_SIZE = 1
NUM_EPOCHS = 160
NUM_CLASSES = 4
NUM_WORKERS = 4
INPUT_SIZE = (128,128,128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print (f"Using device: {DEVICE}")

# Saving the model and other metrics
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/content/drive/MyDrive/4thyearproject_code/RESULTS/SE ENCODER{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Utility
def pad_to_multiple(tensor, multiple=16):
  _,d,h,w = tensor.shape
  pad_d = (multiple - d % multiple) % multiple
  pad_h = (multiple - h % multiple) % multiple
  pad_w = (multiple - w % multiple) % multiple
  return F.pad(tensor, (0,pad_w,0,pad_h,0,pad_d), mode= 'constant', value=0)

class MultimodalDataset(Dataset):
  def __init__(self, patient_dirs, include_mask=True):
      self.patient_dirs = patient_dirs
      self.include_mask = include_mask

  def __len__(self):
    return len(self.patient_dirs)

  def __getitem__(self, index):
      path = self.patient_dirs[index]
      patient_id = os.path.basename(path)
      flair = nib.load(os.path.join(path, f"{patient_id}_flair.nii")).get_fdata()
      t1 = nib.load(os.path.join(path, f"{patient_id}_t1.nii")).get_fdata()
      t1ce = nib.load(os.path.join(path, f"{patient_id}_t1ce.nii")).get_fdata()
      t2 = nib.load(os.path.join(path, f"{patient_id}_t2.nii")).get_fdata()

      image = np.stack([flair, t1, t1ce, t2], axis=0).astype(np.float32)
      image = np.nan_to_num(image)
      for i in range(4):
          max_val = np.max(image[i])
          if max_val > 0:
              image[i] /= max_val

      image_tensor = torch.tensor(image)
      image_tensor = pad_to_multiple(image_tensor)

      if self.include_mask:
          mask = nib.load(os.path.join(path, f"{patient_id}_seg.nii")).get_fdata().astype(np.int64)
          mask[mask == 4] = 3
          mask_tensor = torch.tensor(mask)
          mask_tensor = pad_to_multiple(mask_tensor.unsqueeze(0)).squeeze(0)
          return image_tensor, mask_tensor.unsqueeze(0)
      else:
          return image_tensor

# Addition and integration of SE Block 
class SEBlock(torch.nn.Module):
      def __init__(self, in_channels, reduction_ratio=16):
          super(SEBlock, self).__init__()
          self.avg_pool = torch.nn.AdaptiveAvgPool3d(1)
          self.fc = torch.nn.Sequential(
              torch.nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
              torch.nn.ReLU(inplace=True),
              torch.nn.Linear(in_channels// reduction_ratio, in_channels, bias=False),
              torch.nn.Sigmoid()
          )

       def forward(self, x):
          b, c, _, _, _ = x.size()
          y = self.avg_pool(x).view(b, c)
          y = self.fc(y).view(b, c, 1, 1, 1)
          return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        return self.attn(self.conv(x))

class UNetSEEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, use_se=True):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, channels[0], use_se=True)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(channels[0], channels[1], use_se=True)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(channels[1], channels[2], use_se=True)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock(channels[2], channels[3], use_se=True)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(channels[3], channels[4])

        self.up4 = nn.ConvTranspose3d(channels[4], channels[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(channels[4], channels[3])

        self.up3 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(channels[3], channels[2])

        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(channels[2], channels[1])

        self.up1 = nn.ConvTranspose3d(channels[1],channels[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(channels[1], channels[0])

        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

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


# Loading Data
data_root = "/content/drive/MyDrive/4thyearproject_code/Full_Dataset/Unzipped_File/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
all_dirs = sorted(os.listdir(data_root))
all_dirs = [os.path.join(data_root, d) for d in all_dirs if os.path.isdir(os.path.join(data_root, d))]
train_dirs, val_dirs = train_test_split(all_dirs, test_size=0.2, random_state=42)

train_loader = DataLoader(MultimodalDataset(train_dirs, include_mask=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(MultimodalDataset(val_dirs, include_mask=True), batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# Model Setup
model = UNetSEEncoder(in_channels=4, out_channels=NUM_CLASSES, channels=(16,32,64,128,256), strides=(2,2,2,2), num_res_units=2).to(DEVICE)
pretrained_path = "/content/drive/MyDrive/4thyearproject_code/pretrained_models/model.pt"
if os.path.exists(pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=DEVICE)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

# Encoder freezing 
for name, param in model.named_parameters():
    if name.startswith("enc") or name.startswith("pool"):
        param.requires_grad = False

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
scaler = GradScaler()

# Metrics to be calculated
metrics = {
    'val_loss': [],
    'dice_wt': [], 'dice_tc': [], 'dice_et': [], 'dice_total': [],
    'precision_wt': [], 'precision_tc': [], 'precision_et': [], 'precision_total': [],
    'recall_wt': [], 'recall_tc': [], 'recall_et': [], 'recall_total': [],
    'f1_wt': [], 'f1_tc': [], 'f1_et': [], 'f1_total': [],
    'specificity_wt': [], 'specificity_tc': [], 'specificity_et': []
}

# Training
for epoch in range(1, NUM_EPOCHS + 1):
    if epoch == 10: 
        print("\n Unfreezing encoder layers for fine-tuning")
        for name, param in model.named_parameters():
            if name.startswith("enc") or name.startswith("pool"):
                param.requires_grad = True

    model.train()
    total_loss = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu()
            targets = masks.squeeze(1).cpu()
            all_preds.extend(preds)
            all_targets.extend(targets)

    avg_val_loss = val_loss / len(val_loader)
    metrics['val_loss'].append(avg_val_loss)
    print(f"Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)

    epoch_metrics = compute_all_metrics(all_preds, all_targets)
    for key in ['dice', 'precision', 'recall', 'f1', 'specificity']:
        for sub in ['wt', 'tc', 'et']:
            metrics[f'{key}_{sub}'].append(epoch_metrics[key][sub.upper()])
        if key != 'specificity':
            metrics[f'{key}_total'].append(epoch_metrics[key]['Total'])

    print(f"Metrics — DICE: {epoch_metrics['dice']['Total']:.4f}, F1: {epoch_metrics['f1']['Total']:.4f}, Recall: {epoch_metrics['recall']['Total']:.4f}")

# Saving the model and metrics
torch.save(model.state_dict(), os.path.join(output_dir, f"model1_study2_pretrained_final.pth"))

with open(os.path.join(output_dir, "metrics_summary.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    headers = list(metrics.keys())
    writer.writerow(headers)
    for row in zip(*[metrics[k] for k in headers]):
        writer.writerow(row)
print("Model and metrics saved.")
