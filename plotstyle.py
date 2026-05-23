
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from contextlib import contextmanager
from pathlib import Path

global STYLE_CMAPS 
STYLE_CMAPS = {
    "inputs": "coolwarm",
    "dark": "managua",
    "general": "coolwarm",
    "divergent": "berlin"
}

STYLE_DIR = Path(__file__).parent / "styles"

def _set_cycle_from_cmap(style, n_colors):
    cmap_name = STYLE_CMAPS.get(style, "bone") #default to bone map
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)
    mpl.rcParams["image.cmap"] = cmap_name
    

def use(style, n_colors=8):
    style_path = STYLE_DIR / f"{style}.mplstyle"
    plt.style.use(style_path)
    _set_cycle_from_cmap(style, n_colors)



@contextmanager
def context(style, n_colors=8):
    with mpl.rc_context():

        style_path = STYLE_DIR / f"{style}.mplstyle"
        plt.style.use(style_path)

        _set_cycle_from_cmap(style, n_colors)
        yield