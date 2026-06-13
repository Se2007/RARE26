import numpy as np
import wandb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def compute_metrics(y_true, y_scores):
    """Computes AUROC, AUPRC, PPV@90% Recall, Accuracy, Sensitivity, and Specificity."""

    # AUROC & AUPRC
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    # Compute Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Find PPV @ 90% Recall
    idx = np.where(recalls >= 0.9)[0][-1]
    ppv_at_0_9_recall = precisions[idx]

    # Convert scores to binary predictions (threshold at 0.5)
    y_pred = (y_scores >= 0.5).astype(int)

    # Compute Accuracy, Sensitivity, and Specificity
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    current_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    
    y_pred_at_90_recall = (y_scores >= current_threshold).astype(int)

    tp_at_90 = np.sum((y_pred_at_90_recall == 1) & (y_true == 1))
    tn_at_90 = np.sum((y_pred_at_90_recall == 0) & (y_true == 0))
    fp_at_90 = np.sum((y_pred_at_90_recall == 1) & (y_true == 0))
    fn_at_90 = np.sum((y_pred_at_90_recall == 0) & (y_true == 1))
    
    
    
    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "PPV@90% Recall": ppv_at_0_9_recall,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "T90_Threshold": current_threshold,
        "T90_TP": tp_at_90,
        "T90_FN": fn_at_90,
        "T90_FP": fp_at_90,
        # kept internally so log_wandb_curves can use them — not logged as scalars
        "_y_true": y_true,
        "_y_scores": y_scores,
    }
    return metrics


def log_wandb_curves(metrics: dict, split: str, epoch: int):
    """Logs ROC curve, PR curve, and confusion matrix using compute_metrics output."""

    y_true  = metrics["_y_true"]
    y_scores = metrics["_y_scores"]
    threshold = metrics["T90_Threshold"]

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    wandb.log({
        f"{split}/roc_curve": wandb.plot.line(
            wandb.Table(data=list(zip(fpr.tolist(), tpr.tolist())), columns=["FPR", "TPR"]),
            x="FPR", y="TPR", title=f"{split} ROC Curve"
        )
    }, step=epoch)

    # --- PR Curve ---
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    wandb.log({
        f"{split}/pr_curve": wandb.plot.line(
            wandb.Table(data=list(zip(recalls.tolist(), precisions.tolist())), columns=["Recall", "Precision"]),
            x="Recall", y="Precision", title=f"{split} PR Curve"
        )
    }, step=epoch)

    # --- Confusion Matrix @ T90 threshold ---
    y_pred = (y_scores >= threshold).astype(int)
    wandb.log({
        f"{split}/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=["Negative", "Positive"],
            title=f"{split} Confusion Matrix @ T90"
        )
    }, step=epoch)