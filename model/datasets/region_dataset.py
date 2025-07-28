import torch
import torch.nn.functional as func
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

                for index in range(len(c_list)):
                    d = {
                        "Cities": c_list,
                        "Target": index
                    }
                    self.data.append(d)
        else:
            raise Exception("Not enough cities in region (at least 2 required)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        region = self.data[i]
        inputs, labels, geos = cd.collate_fn(region["Cities"])
        return {
            "Inputs": inputs,
            "Labels": labels,
            "Geo": geos,
            "Target": torch.tensor(region["Target"], dtype=torch.long)
        }


def collate_fn(batch):
    max_cities = max(b["Geo"].shape[0] for b in batch)

    geos_list = []
    inputs_list = []
    labels_list = []
    masks_list = []
    targets_list = []

    for b in batch:
        n_cities = b["Geo"].shape[0]

        geo_padded = func.pad(b["Geo"], (0, 0, 0, max_cities - n_cities))

        inputs_padded = func.pad(b["Inputs"], (0, 0, 0, 0, 0, max_cities - n_cities))

        labels_padded = func.pad(b["Labels"], (0, 0, 0, 0, 0, max_cities - n_cities))

        mask = torch.cat([torch.ones(n_cities), torch.zeros(max_cities - n_cities)])

        geos_list.append(geo_padded)
        inputs_list.append(inputs_padded)
        labels_list.append(labels_padded)
        masks_list.append(mask)
        targets_list.append(b["Target"])

    inputs = torch.stack(inputs_list)
    labels = torch.stack(labels_list)
    geos = torch.stack(geos_list)
    masks = torch.stack(masks_list)
    targets = torch.stack(targets_list)

    return inputs, labels, geos, masks, targets
