import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

data = pd.read_csv("./startinggenomes.csv", index_col = 0)

fig, ax = plt.subplots(figsize=(8,6))
cmap = cm.get_cmap("Dark2").colors
for i, val in enumerate(data.groupby('init_flux')):
    n = val [0]
    grp = val[1]
    ax.scatter(x = "starting fitness", y = "mean", data=grp, label=n, color=cmap[i])
ax.legend()
ax.set_title("Average of initial flux")
ax.set_xlabel("starting fitness")
ax.set_ylabel("end fitness")
plt.show()
