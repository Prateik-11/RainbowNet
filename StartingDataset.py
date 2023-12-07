import torch, os
import pandas as pd

import numpy as np 
import datetime
from datetime import datetime, timedelta 



class StartingDataset(torch.utils.data.Dataset):

    def __init__(self, training_set = True):

        file_path = '/home/prateiksinha/rainbow_final/data/rainbow'
        self.data = np.load(file_path)

        # Get indices that are 1's 
        mask = np.any((self.data[:, 0, :, :] == 1), axis=(1, 2))
        self.rainbow_indices = np.where(mask)[0]


        #print(min(self.rainbow_indices))
        #print(max(self.rainbow_indices))

        mask = np.any((self.data[:, 0, :, :] == 0), axis=(1, 2))
        self.bleak_indices = np.where(mask)[0]

        # Concatenate bleak and rainbow 
        self.ratio = 1 
        self.bleak_indices = self.bleak_indices[:len(self.rainbow_indices) * self.ratio]

        #print(min(self.bleak_indices))
        #print(max(self.bleak_indices))


        self.y_indices = np.concatenate((self.rainbow_indices, self.bleak_indices), axis=0)
        np.random.shuffle(self.y_indices)

        # print(min(self.y_indices))
        # print(max(self.y_indices))

        CONSTANT_SPLIT_RATIO = 0.9
        cutoff = int(len(self.y_indices) * CONSTANT_SPLIT_RATIO)
        if training_set: 
            self.y_indices =  self.y_indices[:cutoff]
        else: 
            self.y_indices =  self.y_indices[cutoff:]

        np.random.shuffle(self.y_indices)
        self.num_imgs = len(self.y_indices) 



        self.rainbow_num = len(self.rainbow_indices)
        #print(f"Number of 1's: {self.rainbow_num}")
        self.bleak_num = len(self.bleak_indices)
        #print(f"Number of 0's: {self.bleak_num}")


        # For our helper function
        self.dt_str = datetime(2004, 1, 1, 0, 0, 0)
        self.dt_end = datetime(2013, 12, 19, 0, 0, 0)

        self.data_root_dir = '/home/prateiksinha/rainbow_final/data/climaX_outputs'
        self.desired_vars = ["u_component_of_wind_100",
                            "u_component_of_wind_250",
                            "u_component_of_wind_500",
                            "u_component_of_wind_1000",

                            "v_component_of_wind_100",
                            "v_component_of_wind_250",
                            "v_component_of_wind_500",
                            "v_component_of_wind_1000",

                            "temperature_100",
                            "temperature_250",
                            "temperature_500",
                            "temperature_1000",

                            "specific_humidity_100",
                            "specific_humidity_250",
                            "specific_humidity_500",
                            "specific_humidity_1000",]

        self.desired_vars_num = len(self.desired_vars)

        print("NUM IMGS IS", self.num_imgs)


    def __getitem__(self, index):

        #assert 0 <= index < self.num_imgs, "Index out of range"

        # First, get the corresponding rainbow_entry 
        idx = self.y_indices[index]
        lbl = self.data[idx].squeeze()

        # Now, let's get the corresponding input data
        entry, npy_file = self.get_npy_file(index) 
        npy_path = os.path.join(self.data_root_dir, npy_file)
        data = np.load(npy_path)

        stacked_arrays = []
        for key in data.files: 
            if key not in self.desired_vars: continue 

            #print(" - Adding key:", key)
            var_data = data[key][entry]
            stacked_arrays.append(var_data)


        #idx_data = np.array(self.desired_vars_num, 128, 256)
        #print(len(stacked_arrays))
        #print(stacked_arrays[0].shape)
        idx_data = np.concatenate(stacked_arrays, axis=0)

        #print("Data shape", idx_data.shape)

        return (idx_data, lbl)



    def __len__(self):
        return self.num_imgs


    def get_npy_file(self, idx): 
        idx_sec = idx * 3600 * 6
        dt_new = self.dt_str + timedelta(seconds=idx_sec)

        year = dt_new.year 
        dt_yr = datetime(year, 1, 1, 0, 0, 0)

        time_elapsed = (dt_new - dt_yr).total_seconds() / 3600
        #print(f"We are {time_elapsed} hours into {year}.")

        if (time_elapsed >= 8736): 
            time_elapsed = 8730

        shard_num = int(time_elapsed / (273))
        entry_num = int(time_elapsed % (273))

        #print(f"So, we're looking for entry {entry_num} in '{year}_{shard_num}.npy'")
        return entry_num, f"{year}_{shard_num}.npz"