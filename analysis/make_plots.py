from pathlib import Path
import matplotlib.pyplot as plt

from common import *

def make_plot(tag):
    W=4
    H=4
    S=1
    fig = plt.figure(figsize=(W*S,H*S))
    ax = plt.axes()
    for arch in RUNS_BY_ARCH:
        zorder = 100 if arch==OUR_ARCH_NAME else 0
        mean, sem = calculate_line(tag, arch)
        mean.plot(x="step", y="value", label=arch, zorder=zorder, ax=ax)
        ax.fill_between(sem.step, mean.value-sem.value, mean.value+sem.value, label=arch, zorder=zorder, alpha=0.25)
    ax.set_xlabel("training samples")
    ax.set_ylabel(tag.split("-")[1])
    ax.set_ylim(0, 1.05)
    plt.savefig(Path("../plots") / Path(tag).with_suffix(".png"), dpi=100)

for tag in TAGS:
    make_plot(tag)
