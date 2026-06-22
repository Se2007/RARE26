import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import sys
from torchvision import transforms

from PIL import Image
import torch
from torch.utils.data import Dataset


# ── constants ──────────────────────────────────────────────────────────────────

CLASS_TO_IDX: Dict[str, int] = {"NDBT": 0, "ACHD": 1}
IDX_TO_CLASS: Dict[int, str] = {0: "NDBT", 1: "ACHD"}

_FNAME_RE = re.compile(r"^pat(\d+)_im(\d+)_(ACHD|NDBT)\.png$", re.IGNORECASE)


# ── dataset ────────────────────────────────────────────────────────────────────

class EVCBarrettsClassification(Dataset):
    """
    Parameters
    ----------
    root_dir  : path to EVC_Barretts_FullSet/ (must contain images/ folder)
    transform : any torchvision-compatible transform applied to the PIL image
    samples   : pre-built list of sample dicts — only used internally by make_splits
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        samples:   Optional[List[Dict]] = None,
    ) -> None:
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = samples if samples is not None else self._scan()

    # ── internal ───────────────────────────────────────────────────────────

    def _scan(self) -> List[Dict]:
        """Walk images/ and build a list of sample dicts."""
        img_dir = self.root_dir / "images"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"'images/' folder not found inside {self.root_dir.resolve()}")

        samples = []
        for path in sorted(img_dir.iterdir()):
            m = _FNAME_RE.match(path.name)
            if m is None:
                continue
            pathology = m.group(3).upper()
            samples.append({
                "path":       path,
                "patient_id": int(m.group(1)),
                "image_idx":  int(m.group(2)),
                "pathology":  pathology,
                "label":      CLASS_TO_IDX[pathology],
            })
        return samples

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s     = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = s["label"]#torch.tensor(s["label"], dtype=torch.long)
        return image, label

    # ── convenience ────────────────────────────────────────────────────────

    @property
    def class_counts(self) -> Dict[str, int]:
        """Number of samples per class."""
        counts = {"NDBT": 0, "ACHD": 0}
        for s in self.samples:
            counts[s["pathology"]] += 1
        return counts

    @property
    def patient_ids(self) -> List[int]:
        return sorted({s["patient_id"] for s in self.samples})

    def __repr__(self) -> str:
        cc = self.class_counts
        return (
            f"EVCBarrettsClassification("
            f"n={len(self)}, "
            f"NDBT={cc['NDBT']}, ACHD={cc['ACHD']}, "
            f"patients={self.patient_ids})"
        )


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    root = sys.argv[1] if len(sys.argv) > 1 else "EVC_Barretts_FullSet"
    print(f"Loading dataset from {root}...")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = EVCBarrettsClassification(root, tf)



    img, label = train_ds.__getitem__(9)
    print(f"\nSample image : {tuple(img.shape)}  label={label.item()} ({IDX_TO_CLASS[label.item()]})")

    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
    plt.show()