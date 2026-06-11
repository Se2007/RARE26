import torch
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric

import numpy as np
import matplotlib.pylab as plt

def train_one_epoch(model, train_loader, loss_fn, optimizer, metric,epoch=None, device='cpu'):
    model.train()
    loss_train = MeanMetric()
    
    all_targets = []
    all_scores = []

    with tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch:
                tepoch.set_description(f'Epoch {epoch}')

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
    
    return model, epoch_metrics



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
  
  return epoch_metrics


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