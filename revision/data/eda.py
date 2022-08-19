import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import seaborn as sns
csv = os.listdir()
csv = [name for name in csv if "csv" in name][0]
csv= pd.read_csv(csv, index_col=[0])
csv['start_fit_round'] = np.round(csv['start_fit'], 2)
csv['end_fit_round'] = np.round(csv['end_fit'], 2)
flux_by_start_fit_round = csv.groupby(['start_flux', 'start_fit_round'], as_index=False)
data = csv[csv['start_flux']!=0.0]
#ax = sns.lineplot(x="start_fit_round", y="end_fit", hue="start_flux", data=data. palette=cm.get_cmap('viridis'))
csv_2 = csv[['start_flux', 'end_fit_round', 'start_fit_round']]
mean = csv_2.groupby(['start_flux', 'start_fit_round'], as_index=False).mean()
sns.lineplot(x='start_flux', y='end_fit_round', hue='start_fit_round', data=mean, palette=colors.Colormap('plasma'))
plt.show()
