from torch.utils.data import Dataset
from pathlib import Path
import torch
import torch.nn.functional as F
import einops
from PIL import Image
import numpy as np

class MoviVideoDataset(Dataset):
    def __init__(self, base_path, half_dimension=True):
        self.video_paths = sorted(Path(base_path).glob("*"))
        self.half_dimension = half_dimension

        if len(self)==0:
            print("!!!!!!!")
            print("WARNING: Dataset is empty! Did you set the correct path?")
            print("!!!!!!!")
            print()
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        frames = []
        for fpath in sorted(self.video_paths[idx].glob("*.png")):
            frame = np.array(Image.open(fpath).convert("RGB"))
            frame = frame / 255.
            frames.append(frame)
        # transpose and stack, list axis is time
        video = einops.rearrange(frames, "listaxis w h c -> listaxis c w h")
        video = torch.Tensor(video)
        if self.half_dimension:
            video = F.avg_pool2d(video, kernel_size=(2,2), stride=(2,2))
        return video
