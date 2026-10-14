"""
setup standard modules to be re-used across notebooks
"""

import torch, numpy as np
import sys, os

"""
NOTE: these next few steps will depend on you having set an environment variable called ATTRRNNS.
You can create this by running the following bash command in terminal:
export ATTRRNNS = PATH-TO-PROJECT
Confirm in bash by just running
echo $ATTRNNS
or in Python:
os.environ["ATTRRNNS"]
There is a fallback option. If the environment is not set
ROOT will default to project path, which will be the cwd
"""
from pathlib import Path
PROJECT_ROOT = Path(
    os.environ.get(
        "ATTRRNNS",
        Path.cwd().parent
    )
)

FIXED_POINTS = PROJECT_ROOT / "FixPointSearch"
STYLES = PROJECT_ROOT / "stylrs"
TASKS = PROJECT_ROOT / "Tasks"
MODELS = PROJECT_ROOT / "saved_models"

for item in [FIXED_POINTS, STYLES, TASKS, MODELS]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from rnn import *
from utils import *
from plot_utils import *
import plotstyle
import matplotlib, matplotlib.pyplot as plt

from IPython import get_ipython
if get_ipython() is not None:
    # run autoreload via setup file
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    get_ipython().run_line_magic("matplotlib", "inline")

from train import train_rnn
import importlib 
from Tasks.TaskManager import Task

"""
NOTE: individual tasks will have to be imported manually
"""

from FixPointSearch.FixedPoint import FixedPoint
from FixPointSearch.fpFinder import *

# standard global variable for input interpolation
# so we don't always have to define it
global NUM_ANGLES
NUM_ANGLES = 25

# standard plotting specs unless otherwise indicated by context
plotstyle.use("inputs", 10)

