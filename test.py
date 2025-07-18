import json
import pandas
import torch
from torch import nn

import utils
from model.weather_network import WeatherNetwork

test_path = "data/Torino/Torino_Weather2025-01-01-2025-07-12.csv"

geo_csv = pandas.read_csv("data/capoluoghi_regione_italiani.csv")

with open('config.json', 'r') as f:
    config = json.load(f)
model_params = config['model_params']

g_row = geo_csv.loc[geo_csv["City"] == "Torino"]
geo = {
    "Latitude": g_row.get("Latitude").item(),
    "Longitude": g_row["Longitude"].item(),
    "Altitude": g_row["Altitude"].item()
}

test_dataset = utils.normalize_and_separate(geo, test_path)
for t in test_dataset[0]["Inputs"]:
    print(utils.denormalize(t[5:]))
    print()

print("-----------PREDICTION-----------\n")
checkpoint = torch.load(model_params['checkpoint'])
hidden_layers = checkpoint['hidden_layers']
dropout = checkpoint['dropout']

model = WeatherNetwork(input_size=10, output_size=5, hidden_layers=hidden_layers, dropout=dropout)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred = model(test_dataset[0]["Geo"].unsqueeze(0), test_dataset[0]["Inputs"].unsqueeze(0))
for p in pred.squeeze(0):
    print(utils.denormalize(p))
    print()

print()
print("-----------REAL VALUES-----------\n")
for r in test_dataset[0]["Labels"]:
    print(utils.denormalize(r))
    print()

loss_fn = nn.SmoothL1Loss()
loss = loss_fn(test_dataset[0]["Labels"], pred.squeeze(0))
print()
print("-----------LOSS-----------")
print(loss.item())

