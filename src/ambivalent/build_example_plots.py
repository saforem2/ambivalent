r"""°°°

# Build Example Plots

![[Open in Colab](https://colab.research.google.com/github/MNoichl/ambivalent/blob/master/build_example_plots.ipynb)](https://colab.research.google.com/assets/colab-badge.svg)

°°°"""
# |%%--%%| <86R7V4Yv0S|3ZNGgtlKOP>
r"""°°°
# Building the example plots

First, we make the standard plot. 
°°°"""
# |%%--%%| <3ZNGgtlKOP|NGT3Gnd9ZL>

# from google.colab import drive
# drive.mount('/content/gdrive/', force_remount=True)


# |%%--%%| <NGT3Gnd9ZL|Wd7ubZRq7l>

# %cd gdrive/MyDrive/ambivalent

# |%%--%%| <Wd7ubZRq7l|v99S6KbmRA>

# ruff: noqa: E402, F401

from typing import Optional
from ambivalent import reload_styles
import seaborn as sns
import ambivalent
from matplotlib import font_manager as fm, pyplot as plt

import os
import shutil
import matplotlib
from pathlib import Path
import pandas as pd

from ambivalent import PROJECT_DIR, STYLES_DIR
OUTPUT_DIR = PROJECT_DIR.joinpath('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

penguins = sns.load_dataset("penguins")

#|%%--%%| <v99S6KbmRA|MCGiYS46iG>

# %load_ext autoreload
# %autoreload 2

#|%%--%%| <MCGiYS46iG|aZfsQfXq22>

def savefig(
        fname: str,
        outdir: Optional[os.PathLike] = None,
):
    outdir = OUTPUT_DIR if outdir is None else outdir
    # fig = plt.gcf()
    ext = {'png', 'svg'}
    print(f"Saving {fname} to {outdir}")
    for ext in ext:
        edir = Path(outdir).joinpath(f"{ext}s")
        edir.mkdir(exist_ok=True, parents=True)
        outfile = Path(edir).joinpath(f"{fname}.{ext}")
        plt.savefig(outfile, dpi=450, bbox_inches='tight')


#|%%--%%| <aZfsQfXq22|O1R7V0rhCm>

import os
import seaborn as sns
from ambivalent import STYLES, set_title_and_suptitle, add_attribution
import colormaps as cmaps

def make_plot(name: str):
    fig, ax = plt.subplots()
    cmap = sns.color_palette(list(cmaps.bold[2:5]._colors))
    sns.scatterplot(
        x="bill_length_mm",
        y="flipper_length_mm",
        data=penguins,
        hue='species',
        style='species',
        palette=cmap,
        alpha=0.75,
    )
    # ambivalent.add_legend()
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
    set_title_and_suptitle(
        "Penguins",
        "An Excellent Bird!"
    )
    savefig(f"penguins_{name}")

#|%%--%%| <O1R7V0rhCm|SmE8QMA9ZI>

plt.style.use('default')
make_plot('default')

#|%%--%%| <SmE8QMA9ZI|nFWJGWKGNm>

reload_styles()


#|%%--%%| <nFWJGWKGNm|jZ6yWZph0U>

with plt.style.context('default'):
    make_plot('default')

for name, stylefile in STYLES.items():
    with plt.style.context(Path(stylefile).as_posix()):
        make_plot(name)
