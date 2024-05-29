# dataset source: https://github.com/HaozhiQi/RPIN/blob/master/docs/PHYRE.md#11-download-our-dataset

from torch.utils.data import Dataset
from pathlib import Path
import utils.fsvisit as fsvisit
import h5py
import torch
import torch.nn.functional as F
import phyre.vis
import einops
import collections



class PhyreVideoDataset(Dataset):
    def _add_hkl_to_list(self, fpath: Path, _):
        if fpath.suffix == ".hkl":
            self.video_paths.append(fpath)

    def __init__(self, base_paths):
        if isinstance(base_paths, str) or not isinstance(base_paths, collections.Collection):
            base_paths = [base_paths]

        self.video_paths = []
        for path in base_paths:
            fsvisit.FSVisitor(
                file_callback=self._add_hkl_to_list
            ).go(Path(path))

        if len(self)==0:
            print("!!!!!!!")
            print("WARNING: Dataset is empty! Did you set the correct path?")
            print("!!!!!!!")
            print()
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
        # .get is the normal dictionary method,
        # [:] is a full slice that has the effect of converting the h5py Dataset into a numpy array (don't know if it's hacky or the legit way)
        f = h5py.File(self.video_paths[idx], 'r')
        video_phyre = f.get("data")[:].astype(int)
        video_phyre = einops.rearrange(video_phyre, "t w h -> w h t")
        video_rgb = phyre.vis.observations_to_float_rgb(video_phyre)
        video_rgb = einops.rearrange(video_rgb, "w h t c -> t c w h")
        video_rgb_tensor = torch.Tensor(video_rgb)
        video_rgb_tensor = F.avg_pool2d(video_rgb_tensor, kernel_size=(2,2), stride=(2,2))
        return video_rgb_tensor



## Assumes identical folder structure after base_video_path and base_label_path
## (which is how the dataset came, btw)
class PhyreVideoLabeledDataset(PhyreVideoDataset):
    def _add_hkl_and_label_to_lists(self, video_path: Path, base_paths):
        if video_path.suffix != ".hkl":
            return
        
        relative_path = video_path.relative_to(base_paths["video"])
        label_path = base_paths["label"] / relative_path
        label_path = label_path.with_name(label_path.name.replace("image", "label"))

        self.video_paths.append(video_path)
        self.label_paths.append(label_path)

    def __init__(self, base_video_paths, base_label_paths):
        if isinstance(base_video_paths, str) or not isinstance(base_video_paths, collections.Collection):
            base_video_paths = [base_video_paths]
        if isinstance(base_label_paths, str) or not isinstance(base_label_paths, collections.Collection):
            base_label_paths = [base_label_paths]
        
        self.video_paths = []
        self.label_paths = []
        for i in range(len(base_video_paths)):
            vpath = base_video_paths[i]
            lpath = base_label_paths[i]
            fsvisit.FSVisitor(
                file_callback=self._add_hkl_and_label_to_lists
            ).go(Path(vpath), {"video": Path(vpath), "label": Path(lpath)})

        if len(self)==0:
            print("!!!!!!!")
            print("WARNING: Dataset is empty! Did you set the correct path?")
            print("!!!!!!!")
            print()
    
    # __len__ can be inherited

    def __getitem__(self, idx):
        video_rgb_tensor = super().__getitem__(idx)
        # as above, but need to use [()] b/c there's a scalar inside
        f = h5py.File(self.label_paths[idx], 'r')
        label = f.get("data")[()]  # an int in {0,1}. 0 is "not solved", 1 is "solved"
        label_tensor = torch.Tensor([label])
        return video_rgb_tensor, label_tensor
