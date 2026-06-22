from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torch.utils.data import Subset, DataLoader, WeightedRandomSampler, ConcatDataset
from sklearn.model_selection import StratifiedKFold, train_test_split

import random
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# from Datasets.rare_dataset import RareDataset
from rare_dataset import RareDataset
from evc_dataset import EVCBarrettsClassification


#---------------------------------------------------------------------#
#                Artifact Removal for Endoscopy Images                #
#---------------------------------------------------------------------#

class EndoscopyArtifactRemover:
    def __init__(self, target_size=(256, 256), black_threshold=20):
        self.target_size = target_size
        self.black_threshold = black_threshold

    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        border = int(min(h, w) * 0.18)
        border_mask = np.zeros_like(gray)

        border_mask[:border, :]  = (gray[:border, :]  < 20) * 255
        border_mask[-border:, :] = (gray[-border:, :] < 20) * 255
        border_mask[:, :border]  = (gray[:, :border]  < 20) * 255
        border_mask[:, -border:] = (gray[:, -border:] < 20) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        border_mask = cv2.dilate(border_mask, kernel, iterations=1)

        result = cv2.inpaint(img_np, border_mask, 7, cv2.INPAINT_TELEA)
        result = cv2.resize(result, self.target_size, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(result)
    
#---------------------------------------------------------------------#
#                 YOLO-style Letterbox Preprocessing                  #
#---------------------------------------------------------------------#
    
class Letterbox:
    def __init__(self, size: int | tuple[int, int] = 224, fill: int = 114):
        self.size = (size, size) if isinstance(size, int) else size
        self.fill = fill 

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size  # PIL → (W, H)
        target_h, target_w = self.size

        scale = min(target_h / h, target_w / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (target_w, target_h), (self.fill,) * 3)
        pad_left = (target_w - new_w) // 2
        pad_top  = (target_h - new_h) // 2
        canvas.paste(img, (pad_left, pad_top))

        return canvas

    def __repr__(self):
        return f"Letterbox(size={self.size}, fill={self.fill})"

#---------------------------------------------------------------------#
#                 Green Channel Contrast Enhancement                  #
#---------------------------------------------------------------------#

class GreenChannelCLAHE:
    def __init__(self, clip_limit=2.0, tile_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)

        g_channel = img_np[:, :, 1]

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_size
        )

        g_enhanced = clahe.apply(g_channel)

        enhanced_rgb = np.stack([g_enhanced, g_enhanced, g_enhanced], axis=-1)

        return Image.fromarray(enhanced_rgb)


#---------------------------------------------------------------------#
#              Multi-Format Dataset Train-Val Splitter                #
#---------------------------------------------------------------------#

def split_dataset(dataset, val_split=0.2, seed=42):
    random.seed(seed)

    neoplasia_indices = []
    ndbe_indices = []

    for i, sample in enumerate(dataset.samples):
        if isinstance(sample, dict): 
            label_str = sample["pathology"].upper()
            if label_str in ["ACHD", "NEOPLASIA"]:
                neoplasia_indices.append(i)
            elif label_str in ["NDBT", "NONDYSPLASTIC"]:
                ndbe_indices.append(i)
        else: 
            label_str = sample[1].lower()
            if label_str in ["neoplasia", "neo"]:
                neoplasia_indices.append(i)
            elif label_str in ["nondysplastic", "ndbe"]:
                ndbe_indices.append(i)

    random.shuffle(neoplasia_indices)
    random.shuffle(ndbe_indices)

    print(len(neoplasia_indices), len(ndbe_indices))

    def split(indices):
        val_size = int(len(indices) * val_split)
        return indices[val_size:], indices[:val_size]

    train_neo, val_neo = split(neoplasia_indices)
    train_ndbe, val_ndbe = split(ndbe_indices)

    train_indices = train_neo + train_ndbe
    val_indices = val_neo + val_ndbe

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

#---------------------------------------------------------------------#
#              RARE Dataset Loader and Class Balancer                 #
#---------------------------------------------------------------------#

class RARE(object):
    def __init__(self, root_path, mode, custom_dataset = RareDataset, transform=None, mini=False, seed=42) :
        assert mode in ['train', 'valid'], 'mode should be train, test or valid'
        self.mini = mini  

        generator  = torch.Generator().manual_seed(seed)
        dataset    = custom_dataset(root_dir=root_path, transform=transform)

        self.train_dataset, self.val_dataset = split_dataset(dataset, seed=seed)


        if mode == 'train' :
            self.dataset = self.train_dataset
        elif mode == 'valid':
            self.dataset = self.val_dataset

        if self.mini:
            mini_size = min(1000, len(self.dataset))
            self.dataset, _ = random_split(
                self.dataset, [mini_size, len(self.dataset) - mini_size], 
                generator=generator
            )

    
    def __call__(self, batch_size, balance_classes=True):
        if balance_classes and self.mode == 'train':
            labels = []
            
            for idx in range(len(self.dataset)):
                _, label = self.dataset[idx]
                labels.append(int(label))
            
            labels = np.array(labels)
            class_counts = np.bincount(labels)
            
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True  # اجازه تکرار تصاویر اقلیت برای ایجاد توازن
            )
            
            data_loader = DataLoader(
                self.dataset, 
                batch_size=batch_size,
                sampler=sampler, 
                num_workers=2,
                pin_memory=True
            )
        else:
            data_loader = DataLoader(
                self.dataset, 
                batch_size=batch_size,
                shuffle=(self.mode == 'train'), 
                num_workers=2,
                pin_memory=True
            )
       
        return data_loader


#---------------------------------------------------------------------#
#              Multi-Dataset Stratified Data Pipeline                 #
#---------------------------------------------------------------------#


def StratifiedKFold_Loader(
    evc_root="EVC_Barretts_FullSet", 
    rare_root="RARE25-train-data",
    use_evc=True, 
    use_rare=True,
    transform=None, 
    batch_size=16, 
    k_fold=False, 
    num_folds=5, 
    balance_classes=True, 
    mini=False, 
    seed=42
):
    
    datasets_to_combine = []
    
    if use_evc:
        evc_ds = EVCBarrettsClassification(root_dir=evc_root, transform=transform)
        datasets_to_combine.append(evc_ds)
        print(f"-> EVC Dataset loaded ({len(evc_ds)} samples)")
        
    if use_rare:
        rare_ds = RareDataset(root_dir=rare_root, transform=transform)
        datasets_to_combine.append(rare_ds)
        print(f"-> RARE Dataset loaded ({len(rare_ds)} samples)")
        
    if len(datasets_to_combine) == 0:
        raise ValueError("")

    base_dataset = ConcatDataset(datasets_to_combine)
    print(f"Total Combined Dataset Size: {len(base_dataset)} samples")

    targets = []
    for ds in base_dataset.datasets:
        for sample in ds.samples:
            if isinstance(sample, dict):  
                targets.append(sample["label"])
            else: 
                targets.append(1 if sample[1] in ["neoplasia", "neo"] else 0)
    targets = np.array(targets)


    if mini:
        mini_size = min(1000, len(targets))
        mini_idx = np.random.RandomState(seed).choice(len(targets), mini_size, replace=False)
        base_dataset = Subset(base_dataset, mini_idx)
        targets = targets[mini_idx]

    indices_pool = np.arange(len(targets))


    def build_loader(indices, labels, is_train):
        subset = Subset(base_dataset, indices)
        
        if is_train and balance_classes:
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
            return DataLoader(subset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
        else:
            return DataLoader(subset, batch_size=batch_size, shuffle=is_train, num_workers=2, pin_memory=True)

    if not k_fold:
        train_idx, val_idx = train_test_split(indices_pool, test_size=0.2, stratify=targets, random_state=seed)
        
        train_loader = build_loader(train_idx, targets[train_idx], is_train=True)
        val_loader = build_loader(val_idx, targets[val_idx], is_train=False)
        
        yield 0, train_loader, val_loader
        
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(indices_pool, targets)):
            train_loader = build_loader(train_idx, targets[train_idx], is_train=True)
            val_loader = build_loader(val_idx, targets[val_idx], is_train=False)
            
            yield fold + 1, train_loader, val_loader




if __name__=='__main__':

    ## Quick test for combined dataset and stratified loader ##

    transform = transforms.Compose([Letterbox(size=224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    loaders_generator = StratifiedKFold_Loader(
        evc_root="EVC_Barretts_FullSet",
        rare_root="RARE25-train-data",
        use_evc=True,        
        use_rare=True,       
        transform=transform,
        batch_size=16,
        k_fold=False,        
        num_folds=5,
        balance_classes=True, 
        mini=False
    )

    for fold, train_loader, val_loader in loaders_generator:
        print(f"\n--- Training Fold {fold} ---")
        # print(len(next(iter(train_loader))))
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"Labels in the very first batch: {labels.tolist()}")
        
        print(f"Train Loader: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"Val Loader:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")




    '''
    ## Quick test for RARE Dataset and Loader ##

    # root_path = "./RARE25-train-data"
    root_path = sys.argv[1] if len(sys.argv) > 1 else "EVC_Barretts_FullSet"

    train_rare = RARE(root_path, mode='train', custom_dataset=EVCBarrettsClassification, mini=True)
    train_loader = train_rare(batch_size=16)

    data_iter = iter(train_loader)

    images, labels = next(data_iter)
    print(f"Batch image shape: {images.shape}, Batch labels shape: {labels.shape}")

    print(f"Image shape: {images[0].shape}, Label: {labels[0]}")

    plt.imshow(images[0].permute(1,2,0))
    plt.axis('off')
    plt.show()'''