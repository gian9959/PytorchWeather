import torch
from torch.utils.data import Dataset

import model.datasets.city_dataset as cd


class RegionWeatherDataset(Dataset):
    def __init__(self, cities):
        if len(cities) >= 2:
            self.data = []
            for i in range(len(cities[0])):
                c_list = []
                for c in cities:
                    c_list.append(c[i])
                self.data.append(c_list)
        else:
            raise Exception("Not enough cities in region (at least 2 required)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        region = self.data[i]
        inputs, labels, geos = cd.collate_fn(region)
        return {
            "Inputs": inputs,
            "Labels": labels,
            "Geo": geos
        }
