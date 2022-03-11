import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

li = []
listfiles = os.listdir('./data/')
for files in listfiles:
    if '.csv' in files:
        li.append(pd.read_csv(f"./data/{files}", index_col=0))

data= pd.concat(li, axis=0, ignore_index=True)


#data = pd.read_csv("./data/data1.csv", index_col = 0)
#data.groupby('init_flux')
#sns.relplot(x="point", y="end_fitness", hue="init_flux")#'line', data=row, palette=sns.color_palette("light:#5A9", as_cmap=True))
x = data.groupby(['init_flux', 'point'], as_index=False)['end_fitness'].mean()

#sns.catplot(x='point', y='end_fitness', hue="init_flux", data=data, kind='box')
ax = sns.relplot(x="point", y="end_fitness", hue="init_flux", data=x, kind='line', markers=True, legend="full", style='init_flux', dashes=False, palette=sns.color_palette("tab20", as_cmap=True))#points = [(x+1)*.25 for x in range(16)]
ax.set(xlabel='Starting point',
       ylabel='End Fitness',
       title='Fluctuation size on starting fitness')
sns.set(rc={"figure.figsize":(3, 4)})
#plt.plot([0.1,1],[0.1,1], color='k')
plt.savefig("./images/init_flux-perf.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
