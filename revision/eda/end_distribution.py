import os
import numpy as np
import matplotlib.pyplot as plt
from visdata import *


folder = os.listdir("../data/")
folder = [name for name in folder if "RPG" in name][0]
os.chdir(f"../data/")
pathname = folder
plotDistributionParam("running_average_performances", show=True, pathname=pathname)
