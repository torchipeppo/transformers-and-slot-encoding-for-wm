import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
from phyre.vis import observations_to_float_rgb, save_observation_series_to_gif

boh = hkl.load("../../../phyre-dataset/PHYRE_1fps_p100n400/full/00022/004/000_image.hkl").astype(int)

boh = np.transpose(boh, [1,2,0])
imgs = observations_to_float_rgb(boh)
imgs = np.transpose(imgs, [2,0,1,3])

for i in range(imgs.shape[0]):
    plt.imshow(imgs[i])
    plt.show()
