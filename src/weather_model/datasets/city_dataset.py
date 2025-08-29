import torch
from torch.utils.data import Dataset


class CityWeatherDataset(Dataset):
    def __init__(self, geo, raw_data, date_size=5, window_size=168, horizon_size=24):
        self.data = []
        geo["Latitude"] = torch.tensor(geo["Latitude"], dtype=torch.float32)
        geo["Longitude"] = torch.tensor(geo["Longitude"], dtype=torch.float32)
        geo["Altitude"] = torch.tensor(geo["Altitude"], dtype=torch.float32)

        in_list = []
        lab_list = []
        for index, row in enumerate(raw_data):
            d = list(row)

            i = int(index) % (window_size + horizon_size)
            if i < window_size:
                d = torch.tensor(d, dtype=torch.float32)
                in_list.append(d)
            else:
                d = torch.tensor(d[date_size:], dtype=torch.float32)
                lab_list.append(d)
                if i == (window_size + horizon_size) - 1:
                    data_dict = dict()
                    in_tensor = torch.stack([t for t in in_list])
                    lab_tensor = torch.stack([l for l in lab_list])
                    geo_tensor = torch.stack([geo["Latitude"], geo["Longitude"], geo["Altitude"]])
                    data_dict.update({"Inputs": in_tensor})
                    data_dict.update({"Labels": lab_tensor})
                    data_dict.update({"Geo": geo_tensor})
                    self.data.append(data_dict)
                    in_list = []
                    lab_list = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(batch):
    inputs = torch.stack([b["Inputs"] for b in batch])
    labels = torch.stack([b["Labels"] for b in batch])
    geos = torch.stack([b["Geo"] for b in batch])

    return inputs, labels, geos
