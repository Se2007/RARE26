import torch
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric

import numpy as np
import matplotlib.pylab as plt

def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None, fold=None, device='cpu'):
    model.train()
    loss_train = MeanMetric()
    
    all_targets = []
    all_scores = []

    desc = "🚀 Training"
    if epoch is not None:
        desc = f"🚀 Epoch {epoch}"
        if fold is not None:
            desc += f" | Fold {fold}"

    with tqdm(train_loader, unit='batch', desc=desc, leave=False) as tepoch:
        for inputs, targets in tepoch:

            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1).float()

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), weight=len(targets))
            
            probs = torch.sigmoid(outputs)
            
            all_targets.append(targets.detach().cpu().numpy())
            all_scores.append(probs.detach().cpu().numpy())

            tepoch.set_postfix(loss=loss_train.compute().item())

    all_targets = np.vstack(all_targets).flatten()
    all_scores = np.vstack(all_scores).flatten()
    
    epoch_metrics = metric(all_targets, all_scores)
    
    epoch_metrics["Loss"] = loss_train.compute().item()
    
    return model, epoch_metrics, all_targets, all_scores



def evaluate(model, test_loader, loss_fn, metric, device='cpu'):
  model.eval()
  loss_eval = MeanMetric()
  
  all_targets = []
  all_scores = []

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device).unsqueeze(1).float()

      outputs = model(inputs)

      loss = loss_fn(outputs, targets)
      loss_eval.update(loss.item(), weight=len(targets))

      probs = torch.sigmoid(outputs)
            
      all_targets.append(targets.detach().cpu().numpy())
      all_scores.append(probs.detach().cpu().numpy())

  all_targets = np.vstack(all_targets).flatten()
  all_scores = np.vstack(all_scores).flatten()
  
  epoch_metrics = metric(all_targets, all_scores)
  
  epoch_metrics["Loss"] = loss_eval.compute().item()
  
  return epoch_metrics, all_targets, all_scores


def load(model, optimizer=None, loss_fn=None, device='cpu', reset=False, load_path=None):
    if reset == False and load_path is not None:
        state = torch.load(load_path, map_location=torch.device(device))
        
        model.load_state_dict(state['state_dict'])
        model = model.to(device)
        
        if optimizer is not None and 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
            optimizer_to(optimizer, device)
        
        if loss_fn is not None and 'loss_fun' in state and hasattr(loss_fn, 'load_state_dict'):
            loss_fn.load_state_dict(state['loss_fun'])
            print("Loaded learnable parameters for Loss Function.")
            
        if 'epoch' in state:
            print(f"Loaded checkpoint from epoch {state['epoch']}")
            
    else:
        model = model.to(device)
        if load_path is None and reset == False:
            print('Warning: give path for load model')

    return model, loss_fn, optimizer

def optimizer_to(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save(save_path, model, optimizer, loss_fn=None, epoch=None):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    if loss_fn is not None and hasattr(loss_fn, 'state_dict'):
        loss_dict = loss_fn.state_dict()
        if loss_dict: 
            state['loss_fun'] = loss_dict
            print("..:: Learnable Loss Function parameters included in checkpoint ::..")
            
    if epoch is not None:
        state['epoch'] = epoch

    torch.save(state, save_path)
    print(f"Checkpoint successfully saved to {save_path}")


def plot(train_hist, valid_hist, label):
    print(f'\nTrained {len(train_hist)} epochs')

    plt.plot(range(len(train_hist)), train_hist, 'k-', label="Train")
    plt.plot(range(len(valid_hist)), valid_hist, 'y-', label="Validation")

    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.legend()
    plt.show()


def create_wandb_log_dict(mean_train_metrics, mean_valid_metrics, get_wandb_curves_fn):
    skip_keys = {"Accuracy", "Sensitivity", "Specificity", "T90_Threshold", "T90_TP", "T90_FP", "T90_FN"}
    
    wandb_log_dict = {
        "Loss/train":         mean_train_metrics["Loss"],
        "Loss/val":           mean_valid_metrics["Loss"],
        "AUROC/train":        mean_train_metrics["AUROC"],
        "AUROC/val":          mean_valid_metrics["AUROC"],
        "AUPRC/train":        mean_train_metrics["AUPRC"],
        "AUPRC/val":          mean_valid_metrics["AUPRC"],
        "PPV@90Recall/train": mean_train_metrics["PPV@90% Recall"],
        "PPV@90Recall/val":   mean_valid_metrics["PPV@90% Recall"],
    }

    for key, value in mean_train_metrics.items():
        if key.startswith("_") or key in skip_keys or key == "Loss" or isinstance(value, np.ndarray):
            continue
        wandb_log_dict[f"train/{key}"] = value

    for key, value in mean_valid_metrics.items():
        if key.startswith("_") or key in skip_keys or key == "Loss" or isinstance(value, np.ndarray):
            continue
        wandb_log_dict[f"val/{key}"] = value
        
    wandb_log_dict.update(get_wandb_curves_fn(mean_train_metrics, "train"))
    wandb_log_dict.update(get_wandb_curves_fn(mean_valid_metrics, "val"))

    return wandb_log_dict