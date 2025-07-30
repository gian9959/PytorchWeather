import json
import random

import pandas
import torch
from torch.utils.data import DataLoader, Subset

import utils
from model.forecast_model import ForecastModel
import model.datasets.region_dataset as rd

test_date = ["2025-01-01-2025-07-12"]

geo_csv = pandas.read_csv("../data/italian_cities.csv")
test_cities = ["Alessandria",
               "Asti",
               "Biella",
               "Cuneo",
               "Novara",
               "Torino",
               "Verbania",
               "Vercelli"]

with open('../config.json', 'r') as f:
    config = json.load(f)
model_params = config['model_params']

test_dataset = utils.load_all(test_date, "../test_data")

INDEX = random.randint(0, len(test_dataset))

# for td in test_dataset[INDEX]["Inputs"]:
#     for t in td:
#         print(utils.denormalize(t[5:]))
#         print()


print()
print(f"CITY: {test_cities[test_dataset[INDEX]['Target']]}")
print("-----------PREDICTION-----------\n")
checkpoint = torch.load(model_params['checkpoint'])
hidden_layers = checkpoint['hidden_layers']
dropout = checkpoint['dropout']

test_subset = Subset(test_dataset, [INDEX])
test_loader = DataLoader(test_subset, batch_size=1, collate_fn=rd.collate_fn, shuffle=False)

model = ForecastModel(input_size=11, output_size=6, hidden_layers=hidden_layers, dropout=dropout)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

for inputs, labels, geos, mask, targets in test_loader:
    pred = model(geos, inputs, targets, mask)

    pred_list = []
    for p in pred.squeeze(0):
        pr = utils.denormalize(p)
        # print(pr)
        pred_list.append(pr)

print(f"TEMPERATURE: {min(list(temp['TEMPERATURE'] for temp in pred_list))} - {max(list(temp['TEMPERATURE'] for temp in pred_list))}")
print(f"AVG HUMIDITY: {sum(list(temp['HUMIDITY'] for temp in pred_list)) / 24}")
print(f"AVG PRECIPITATION: {sum(list(temp['PRECIPITATION'] for temp in pred_list)) / 24}")
print(f"AVG PRESSURE: {sum(list(temp['PRESSURE'] for temp in pred_list)) / 24}")
print(f"AVG CLOUD: {sum(list(temp['CLOUD'] for temp in pred_list)) / 24}")
print(f"AVG WIND: {sum(list(temp['WIND'] for temp in pred_list)) / 24}")

print()
print("-----------REAL VALUES-----------\n")
real_list = []
for inputs, labels, geos, mask, targets in test_loader:
    lab = labels[torch.arange(labels.size(0)), targets]
    for l in lab.squeeze(0):
        real = utils.denormalize(l)
        # print(real)
        real_list.append(real)

print(f"TEMPERATURE: {min(list(temp['TEMPERATURE'] for temp in real_list))} - {max(list(temp['TEMPERATURE'] for temp in real_list))}")
print(f"AVG HUMIDITY: {sum(list(temp['HUMIDITY'] for temp in real_list)) / 24}")
print(f"AVG PRECIPITATION: {sum(list(temp['PRECIPITATION'] for temp in real_list)) / 24}")
print(f"AVG PRESSURE: {sum(list(temp['PRESSURE'] for temp in real_list)) / 24}")
print(f"AVG CLOUD: {sum(list(temp['CLOUD'] for temp in real_list)) / 24}")
print(f"AVG WIND: {sum(list(temp['WIND'] for temp in real_list)) / 24}")
