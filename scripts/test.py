import json
import pandas
import torch
from torch import nn

import utils
from model.weather_network import WeatherNetwork

INDEX = 15

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
for t in test_dataset[INDEX]["Inputs"]:
    print(utils.denormalize(t[5:]))
    print()

print("-----------PREDICTION-----------\n")
checkpoint = torch.load(model_params['checkpoint'])
hidden_layers = checkpoint['hidden_layers']
dropout = checkpoint['dropout']

model = WeatherNetwork(input_size=10, output_size=5, hidden_layers=hidden_layers, dropout=dropout)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred = model(test_dataset[INDEX]["Geo"].unsqueeze(0), test_dataset[INDEX]["Inputs"].unsqueeze(0))
pred_list = []
for p in pred.squeeze(0):
    pr = utils.denormalize(p)
    print(pr)
    pred_list.append(pr)
    print()

print()
print("-----------REAL VALUES-----------\n")
real_list = []
for r in test_dataset[INDEX]["Labels"]:
    real = utils.denormalize(r)
    print(real)
    real_list.append(real)
    print()

loss_fn = nn.SmoothL1Loss()
loss = loss_fn(test_dataset[INDEX]["Labels"], pred.squeeze(0))
print()
print("-----------LOSS-----------")
print(loss.item())

print()
print("-----------SUMMARY-----------")
real_list = pandas.DataFrame.from_dict(real_list)
pred_list = pandas.DataFrame.from_dict(pred_list)
print()
print("PRED:")
print(f"Temp: {min(pred_list['TEMP'])} - {max(pred_list['TEMP'])}")
print(f"Hum: {min(pred_list['HUM'])} - {max(pred_list['HUM'])}")
print(f"Total prec: {pred_list['PREC'].sum}")
print(f"Wind: {min(pred_list['WIND'])} - {max(pred_list['WIND'])}")

print()
print("REAL:")
print(f"Temp: {min(real_list['TEMP'])} - {max(real_list['TEMP'])}")
print(f"Hum: {min(real_list['HUM'])} - {max(real_list['HUM'])}")
print(f"Total prec: {real_list['PREC'].sum}")
print(f"Wind: {min(real_list['WIND'])} - {max(real_list['WIND'])}")
