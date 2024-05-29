"""

"""

from pathlib import Path
from timeit import timeit
import logging

import hickle as hkl
import h5py

class FSVisitor:
    def __init__(self, *, directory_callback=None, file_callback=None):
        self.directory_callback = directory_callback
        self.file_callback = file_callback

    def _visit(self, dir_path, extra_data):
        if self.directory_callback:
            retval = self.directory_callback(dir_path, extra_data)
            if retval:
                extra_data = retval
        for child in dir_path.iterdir():
            self.go(child, extra_data)

    def _act(self, file_path, extra_data):
        if self.file_callback:
            self.file_callback(file_path, extra_data)

    def go(self, path: Path, extra_data = None):
        path.resolve()
        if path.is_dir():
            self._visit(path, extra_data)
        else:
            self._act(path, extra_data)

class MinimalPhyreVideoDataset():
    def _add_hkl_to_list(self, fpath: Path, _):
        if fpath.suffix == ".hkl":
            self.video_paths.append(fpath)

    def __init__(self, base_path, FIXED):
        self.video_paths = []
        FSVisitor(
            file_callback=self._add_hkl_to_list
        ).go(Path(base_path))

        self.FIXED = FIXED

        if len(self)==0:
            print("!!!!!!!")
            print("WARNING: Dataset is empty! Did you set the correct path?")
            print("!!!!!!!")
            print()
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_idx = self.video_paths[idx]

        if self.FIXED:
            f = h5py.File(video_idx, 'r')
            # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
            # .get is the normal dictionary method,
            # [:] is a full slice that has the effect of converting the h5py Dataset into a numpy array (don't know if it's hacky or the legit way)
            video_phyre = f.get("data")[:]
        else:
            video_phyre = hkl.load(video_idx)

        return video_phyre

class MinimalPhyreVideoLabeledDataset(MinimalPhyreVideoDataset):
    def _add_hkl_and_label_to_lists(self, video_path: Path, _):
        if video_path.suffix != ".hkl":
            return
        
        relative_path = video_path.relative_to(self.base_video_path)
        label_path = self.base_label_path / relative_path
        label_path = label_path.with_name(label_path.name.replace("image", "label"))

        self.video_paths.append(video_path)
        self.label_paths.append(label_path)

    def __init__(self, base_video_path, base_label_path, FIXED):
        self.base_video_path = base_video_path
        self.base_label_path = base_label_path
        self.video_paths = []
        self.label_paths = []
        FSVisitor(
            file_callback=self._add_hkl_and_label_to_lists
        ).go(Path(base_video_path))

        self.FIXED = FIXED

        if len(self)==0:
            print("!!!!!!!")
            print("WARNING: Dataset is empty! Did you set the correct path?")
            print("!!!!!!!")
            print()
    
    # __len__ can be inherited

    def __getitem__(self, idx):
        video_phyre = super().__getitem__(idx)
        label_idx = self.label_paths[idx]

        if self.FIXED:
            f = h5py.File(label_idx, 'r')
            # as above, but need to use [()] b/c there's a scalar inside
            label = f.get("data")[()]
        else:
            label = hkl.load(label_idx)

        return video_phyre, label

from omegaconf import OmegaConf
path_constants = OmegaConf.load("../config/path_constants/path_constants.yaml")

def task_factory(dataset, range_idx, range_len):
    def f():
        start = range_idx * range_len
        for i in range(start, start+range_len):
            x, y = dataset[i]
    return f

logging.basicConfig(filename='times.log', level=logging.INFO, format="")

video_path = path_constants.phyre_video_dataset
label_path = video_path.replace("/full/", "/labels/")
dataset = MinimalPhyreVideoLabeledDataset(video_path, label_path, FIXED=True)

i=0
for epoch_idx in range(100):
    for batch_idx in range(10):
        print(f"ep={epoch_idx} ba={batch_idx}")
        t = timeit(task_factory(dataset, i, 50), number=1)
        print(f"time={t}")
        logging.info(f"ep={epoch_idx}  ba={batch_idx}  t={t}")
        print()
        i+=1
