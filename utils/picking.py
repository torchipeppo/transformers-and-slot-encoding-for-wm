from pathlib import Path
from dataset import PhyreVideoDataset, PhyreVideoLabeledDataset, MoviVideoDataset

def pick_dataset(conf):
    if conf.dataset.dataset.lower() == "phyre":
        return PhyreVideoDataset(conf.path_constants.phyre_video_dataset)
    if conf.dataset.dataset.lower() == "phyre-labeled":
        video_paths = conf.path_constants.phyre_video_dataset
        label_paths = [p.replace("/full/", "/labels/") for p in video_paths]  # assuming folders are named as given by the source of the dataset
        return PhyreVideoLabeledDataset(video_paths, label_paths)
    elif conf.dataset.dataset.lower() == "movi-e":
        return MoviVideoDataset(conf.path_constants.movi_e_video_dataset)
    else:
        raise RuntimeError(f'Unknown dataset conf "{conf.dataset.dataset}"')

def pick_pretrained_tokenizer_path(conf):
    if conf.dataset.dataset.lower() in {"phyre", "phyre-labeled"}:
        return conf.path_constants.phyre_pretrained_tokenizer
    elif conf.dataset.dataset.lower() == "movi-e":
        return conf.path_constants.movi_e_pretrained_tokenizer
    else:
        raise RuntimeError(f'Unknown dataset conf "{conf.dataset.dataset}"')

def pick_pretrained_3transf_path(conf):
    if conf.dataset.dataset.lower() in {"phyre", "phyre-labeled"}:
        return conf.path_constants.phyre_pretrained_3transf
    elif conf.dataset.dataset.lower() == "movi-e":
        return conf.path_constants.movi_e_pretrained_3transf
    else:
        raise RuntimeError(f'Unknown dataset conf "{conf.dataset.dataset}"')

def pick_pretrained_decodonly_path(conf):
    if conf.dataset.dataset.lower() in {"phyre", "phyre-labeled"}:
        return conf.path_constants.phyre_pretrained_decodonly
    elif conf.dataset.dataset.lower() == "movi-e":
        return conf.path_constants.movi_e_pretrained_decodonly
    else:
        raise RuntimeError(f'Unknown dataset conf "{conf.dataset.dataset}"')

def pick_pretrained_steve_path(conf):
    if conf.dataset.dataset.lower() in {"phyre", "phyre-labeled"}:
        return conf.path_constants.phyre_pretrained_steve
    elif conf.dataset.dataset.lower() == "movi-e":
        return conf.path_constants.movi_e_pretrained_steve
    else:
        raise RuntimeError(f'Unknown dataset conf "{conf.dataset.dataset}"')
