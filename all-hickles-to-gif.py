"""
Search recursively the given directory for all PHYRE .hkl files and convert them to GIFs for visualization
"""

import sys
from pathlib import Path
import hickle as hkl
from phyre.vis import save_observation_series_to_gif
import utils

def visit(the_dir, parent_gif_dir):
    the_dir = the_dir.resolve()
    gif_dir = parent_gif_dir / (the_dir.name+"_gif")
    if not gif_dir.exists():
        gif_dir.mkdir()
    return gif_dir

def gifify(hkl_path, gif_dir):
    if hkl_path.suffix != ".hkl":
        return
    assert "full" in str(hkl_path)

    label_dir_path = Path(str(hkl_path).replace("full", "labels")).parent
    label_path = label_dir_path / (hkl_path.name.replace("image", "label"))

    gif_name = hkl_path.with_suffix(".gif").name
    if (gif_dir / gif_name).exists():
        # print(f"{gif_dir / gif_name} already exists, skipping")
        pass
    else:
        obsseries = hkl.load(hkl_path).astype(int)
        label = hkl.load(label_path)  # an int in {0,1}. 0 is "not solved", 1 is "solved"
        # gotta give lists b/c this expects "batches"
        save_observation_series_to_gif([obsseries], gif_dir / gif_name, [label])


# main

start = Path(sys.argv[1]).resolve()
utils.FSVisitor(
    directory_callback=visit,
    file_callback=gifify
).go(start, start.parent)
