import torch
from torch import nn
from torch import optim

from Datasets.dataset import RARE
from methods.ResNet import resnet50

from utils import train_one_epoch
from metrics import compute_metrics
from loss import WeightedFocalLoss, HybridBCEMedicalLoss

from prettytable import PrettyTable
from colorama import Fore, Style, init

## Function for load the model for change and find the hyperparameter during the training

def load(model, device='cpu', reset = False, load_path = None):
    model = model

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
    return model

####  Arguments

device = 'cuda'
num_epochs = 5
reset = True
root_path = "./Datasets/RARE25-train-data"

train_loader = RARE(root_path, mode='train', mini=True)(batch_size=64)

load_path = './saved_model/' + '' + ".pth"

#######################
#   Hyperparameters   #
#######################

learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003]
weight_decays = [1e-2, 1e-4, 1e-6]

## preprocessing for makeing the table and finding the minimums

loss_list = []

best_lr = None
best_wd = None
best_loss = float('inf')  
min_num = float('inf')
second_min = float('inf')

table = PrettyTable()
table.field_names = ["LR \ WD"] + [f"WD {i}" for i in weight_decays]

## Loss function and Metric

metric = compute_metrics

pos_weight_tensor = torch.tensor([10.0], dtype=torch.float32)

# loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = WeightedFocalLoss(alpha=0.95, gamma=1.5).to(device)
# loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
loss_fn = HybridBCEMedicalLoss(
    pos_weight_bce=20.0,  
    alpha_focal=0.95,     
    gamma_focal=1.0,      
    lambda_hybrid=0.5      
).to(device)


for lr in learning_rates:
    for wd in weight_decays:
    
        print(f'\nLR={lr}, WD={wd}')

        ## Model and Optimizer

        model = resnet50(pretrained=True).to(device)

        ### Calculate the amount of parameters
        print(sum(p.numel() for p in model.parameters()))
        
        model = load(model, device=device, reset = reset, load_path = load_path)
        
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=False)


        for epoch in range(1, num_epochs+1):
            model, epoch_metrics, _, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch, device=device)

        print(f"Final Loss: {epoch_metrics['Loss']}")   
     
        loss_list.append(float(f'{epoch_metrics["Loss"]:.4f}'))

## Add the color to the first and second minimun loss of the table

sorted_list = sorted(loss_list)
first_min = sorted_list[0]
second_min = sorted_list[1]

first_min_idx = loss_list.index(first_min)
second_min_idx = loss_list.index(second_min)

loss_list[first_min_idx] = f"{Fore.GREEN}{first_min}{Fore.WHITE}"
loss_list[second_min_idx] = f"{Fore.YELLOW}{second_min}{Fore.WHITE}"
loss_list = list(map(str, loss_list))

## Making the table

o = 0

for i in learning_rates:
    row = [f"LR {i}"]

    losses = loss_list[o:len(weight_decays)+o]
    o += len(weight_decays)

    row += losses
    table.add_row(row)


print(table)