import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
folders = os.listdir("./data")
folder = [name for name in folders if "iter" in name][0]
#os.chdir(f"./data/{folder}")
folders = os.listdir(f"./data/{folder}")
folders = [name for name in folders if "csv" not in name and 'txt' not in name]
data = ""
trial_name = []
folder_name = []
start_fitness = []
end_fitness = []
begin_reward = []
end_reward = []
init_flux =[]
end_flux = []
min_period = []
max_period = []

for name in folders:
    files = os.listdir(f"./data/{folder}/{name}")
    files = [n for n in files if "npz" in n]
    print(f"Number of files for {name}:{len(files)}")
    mn_pd = [attr.split(".")[1] for attr in name.split("_") if "mnP" in attr][0]
    mx_pd = [attr.split(".")[1] for attr in name.split("_") if "mxP" in attr][0]
    for filename in files:
        data = np.load(f"./data/{folder}/{name}/{filename}")
        begin_reward.append(data['reward'][0])
        min_period.append(mn_pd)
        max_period.append(mx_pd)
        end_reward.append(data['reward'][-1])
        trial_name.append(f"{filename}")
        folder_name.append(f"{name}")
        start_fitness.append(data['start_fitness'])
        end_fitness.append(data['end_fitness'])
        init_flux.append(data['init_flux'])
        end_flux.append(data['flux_amp'][-1])
rows = {"trial_name": trial_name, "start_fit":start_fitness, "end_fit":end_fitness,
        "start_reward":begin_reward, "end_reward":end_reward,
        "start_flux":init_flux, "end_flux":end_flux, "folder_name":folder_name, "min_period":min_period, "max_period":max_period}
csv = pd.DataFrame(rows)
csv.to_csv(f"./data/{folder}/summary_data.csv")
