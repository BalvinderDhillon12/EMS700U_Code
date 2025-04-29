# This code is to compute the metrics for all of the studies. The values are then compiled into a csv file which will 
# then be used to visualise it. The raw data can be found in one of the folders here or in the link attached in the Appendix.

import torch
import numpy as np
from scipy import ndimage
from medpy.metric import binary
import seaborn as sns 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def compute_dice(pred, target, class_idx): # this checks the overlap between ground truth and predicted 
    pred_c = (pred == class_idx).float()
    target_c = (target == class_idx).float()
    inter = (pred_c * target_c).sum().item()
    total = pred_c.sum().item() + target_c.sum().item()
    return 2 * inter / total if total > 0 else 1.0

def compute_precision(pred, target, class_idx): # this calculates the rate of positive predictions
    tp = ((pred == class_idx) & (target == class_idx)).sum().item()
    fp = ((pred == class_idx) & (target != class_idx)).sum().item()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def compute_recall(pred, target, class_idx): # this computes all the true positives and false negatives within a class
    tp = ((pred == class_idx) & (target == class_idx)).sum().item()
    fn = ((pred != class_idx) & (target == class_idx)).sum().item()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def compute_specificity(pred, target, class_idx): # this computes the rate of negative but true predictions
    tn = ((pred != class_idx) & (target != class_idx)).sum().item()
    fp = ((pred == class_idx) & (target != class_idx)).sum().item()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def compute_f1(precision, recall): # this is used to evaluate the classification of each subregion
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# this calculates the overlap of the margins between each subregion 
def compute_boundary_dice(pred, target, class_idx, margin=2): 
    pred_np = (pred == class_idx).cpu().numpy()
    target_np = (target == class_idx).cpu().numpy()

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

# This computes the model's confusion between different classes. due to runtime limits, the confusion matrix was calculated seperately 
# by retraining the model. All the code remains the same as in the studies but instead of computing all the metrics, only the confusion metric 
# was calculated.
    
def compute_confusion_matrix (preds,targets): 
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(targets_flat, preds_flat):
        confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

def compute_all_metrics(preds, targets):
    all_metrics = {
        'dice': {},
        'precision': {},
        'recall': {},
        'f1': {},
        'specificity': {},
    }
    classes = {'WT': 1, 'TC': 2, 'ET': 3}  # label mapping
    
    preds = torch.cat([p.unsqueeze(0) for p in preds], dim=0)
    targets = torch.cat([t.unsqueeze(0) for t in targets], dim=0)
    
    for region, class_idx in classes.items():
        dsc = compute_dice(preds, targets, class_idx)
        prec = compute_precision(preds, targets, class_idx)
        rec = compute_recall(preds, targets, class_idx)
        f1 = compute_f1(prec, rec)
        spec = compute_specificity(preds, targets, class_idx)

        all_metrics['dice'][region] = dsc
        all_metrics['precision'][region] = prec
        all_metrics['recall'][region] = rec
        all_metrics['f1'][region] = f1
        all_metrics['specificity'][region] = spec
    confusion_matrix = compute_confusion_matrix(preds, targets, num_classes=4)
    
    return all_metrics, confusion_matrix 
