from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

import random
import torch
from torch.utils.data import DataLoader, Subset, random_split

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class RareDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the main dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else self.default_transforms()
        self.classes = {"neo": "neoplasia", "ndbe": "nondysplastic"}
        self.class_counts = {"neoplasia": 0, "nondysplastic": 0}
        self.samples = self.load_samples()

        # Print counts after loading
        print(
            f"Loaded dataset with {self.class_counts['neoplasia']} 'neo' (neoplasia) images and "
            f"{self.class_counts['nondysplastic']} 'ndbe' (nondysplastic) images."
        )

    def default_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),

                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

    def load_samples(self):
        samples = []
        for center in self.root_dir.iterdir():
            if center.is_dir():
                for class_folder in ["neo", "ndbe"]:
                    class_dir = center / class_folder
                    if class_dir.exists():
                        for img_path in class_dir.glob("*.png"):
                            label = self.classes[class_folder]
                            samples.append((img_path, label))
                            self.class_counts[label] += 1  # Count the sample
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = 1 if label == "neoplasia" else 0
        return image, label
    


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


def split_dataset(dataset, val_split=0.2, seed=42):
    random.seed(seed)

    # Separate indices by string label
    neoplasia_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == "neoplasia"
    ]
    ndbe_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == "nondysplastic"
    ]

    # Shuffle both lists
    random.shuffle(neoplasia_indices)
    random.shuffle(ndbe_indices)

    # Split function
    def split(indices):
        val_size = int(len(indices) * val_split)
        return indices[val_size:], indices[:val_size]  # train, val

    # Split each class
    train_neo, val_neo = split(neoplasia_indices)
    train_ndbe, val_ndbe = split(ndbe_indices)

    # Combine splits
    train_indices = train_neo + train_ndbe
    val_indices = val_neo + val_ndbe

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)



class RARE(object):
    def __init__(self, root_path, mode, transform=None, mini=False, seed=42) :
        assert mode in ['train', 'valid'], 'mode should be train, test or valid'
        self.mini = mini  

        generator  = torch.Generator().manual_seed(seed)
        dataset    = RareDataset(root_dir=root_path, transform=transform)

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


    def __call__(self, batch_size) :

        data_loader = DataLoader(
                                self.dataset, 
                                batch_size=batch_size,
                                shuffle=True, 
                                # num_workers=0,
                                # pin_memory=True 
                                )
       


        return data_loader




if __name__=='__main__':
    root_path = "./RARE25-train-data"

    """
    custom_dataset = RareDataset(root_dir=root_path, transform=None)
    print(f"Total samples in dataset: {len(custom_dataset)}")
    print(f"Class distribution: {custom_dataset.class_counts}")

    image, label = custom_dataset.__getitem__(1125)
    print(f"Image shape: {image.shape}, Label: {label}")

    plt.imshow(image.permute(1,2,0))
    plt.axis('off')
    plt.show()
    """

    train_rare = RARE(root_path, mode='train', mini=True)
    train_loader = train_rare(batch_size=16)

    data_iter = iter(train_loader)

    images, labels = next(data_iter)
    print(f"Batch image shape: {images.shape}, Batch labels shape: {labels.shape}")

    print(f"Image shape: {images[0].shape}, Label: {labels[0]}")

    plt.imshow(images[0].permute(1,2,0))
    plt.axis('off')
    plt.show()