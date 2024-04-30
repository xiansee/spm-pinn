import os
import pickle

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from core.physics_model import SPM


class SimulationDataset(Dataset):
    norm_settings = {
        SPM.time_col: (0, 50),
        SPM.rp_col: (0, 6e-6),
        SPM.rn_col: (0, 6e-6),
    }

    def __init__(self, data_directory: list) -> None:
        self.data = [
            self.get_data(f"{data_directory}/{file_path}")
            for file_path in os.listdir(data_directory)
        ]

    def get_data(self, file_path: str) -> dict:
        with open(file_path, "rb") as binary_file:
            data = pickle.load(binary_file)

        data = self.normalize_data(data)
        t, rp = np.meshgrid(data[SPM.time_col], data[SPM.rp_col][0])
        t, rn = np.meshgrid(data[SPM.time_col], data[SPM.rn_col][0])

        I = Tensor(data[SPM.current_col]).view(len(data[SPM.time_col]), 1)
        Xp = torch.stack([Tensor(t.flatten()), Tensor(rp.flatten())], axis=-1)
        Xn = torch.stack([Tensor(t.flatten()), Tensor(rn.flatten())], axis=-1)
        Y = Tensor(data[SPM.voltage_col]).view(len(data[SPM.time_col]), 1)

        N_t = len(data[SPM.time_col])
        N_rp = len(data[SPM.rp_col][0])
        N_rn = len(data[SPM.rn_col][0])

        Xp.requires_grad = True
        Xn.requires_grad = True

        return I, Xp, Xn, Y, (N_t, N_rp, N_rn)

    def normalize_data(self, data: dict):
        for col, norm_range in self.norm_settings.items():
            min_val, max_val = norm_range
            data[col] = np.array(data[col])
            data[col] = (data[col] - min_val) / (max_val - min_val) * 2 - 1

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor]:
        return self.data[index]
