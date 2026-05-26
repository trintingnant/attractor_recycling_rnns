
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from contextlib import contextmanager
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

#Create custom color maps for plotting contexts:

CUSTOM_CMAPS = {
    "hotpink_grey": [
        (0.0, "hotpink"),   
        (0.5, "white"),     
        (1.0, "darkgrey"), 
    ],
    
    "mint_darkgreen": [
        (0.0, "#98ffcc"),
        (0.5, "white"), 
        (1.0, "#004d1a"),    
    ]
   
}

STYLE_CMAPS = {
    "inputs": "coolwarm",
    "dark": "managua",
    "general": "hotpink_grey",
    "divergent": "berlin"
}

STYLE_DIR = Path(__file__).parent / "styles"

for name, stops in CUSTOM_CMAPS.items():
    cmap = LinearSegmentedColormap.from_list(
        name,
        stops,
        N=256
    )

    mpl.colormaps.register(cmap, force=True)

def _get_cmap(name):
    return mpl.colormaps.get_cmap(name)


def _set_cycle_from_cmap(style, n_colors):
    cmap_name = STYLE_CMAPS.get(style, "bone") #default to bone map
    cmap = _get_cmap(cmap_name)
    print(type(cmap))
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





