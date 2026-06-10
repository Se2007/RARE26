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

'''
def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None, device='cpu'):
  model.train()
  loss_train = MeanMetric()
#   metric.reset()

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
      #   metric.update(outputs, targets.to(torch.int32))

      tepoch.set_postfix(loss=loss_train.compute().item(),
                        #  metric=metric.compute().item()
                         )

  return model, loss_train.compute().item()#, metric.compute().item()
'''

def evaluate(model, test_loader, loss_fn, metric, device='cpu'):
  model.eval()
  loss_eval = MeanMetric()
  metric.reset()

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device).unsqueeze(1)

      outputs = model(inputs)

      loss = loss_fn(outputs, targets)
      loss_eval.update(loss.item(), weight=len(targets))

      metric(outputs, targets.to(torch.int32))

  return loss_eval.compute().item(), metric.compute().item()