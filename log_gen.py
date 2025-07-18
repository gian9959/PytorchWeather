import os
import json
import pandas
import torch
from torch.utils.data import DataLoader

import utils
import learningFunctions as lf
from model.weatherDataset import collate_fn

val_files = ["2020-01-01-2024-12-31"]

with open('config.json', 'r') as f:
    config = json.load(f)

print('Loading validation dataset...')
val_dataset = utils.load_all(val_files)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

epochs = []
tr_loss = []
val_loss = []

log_params = config['log_params']

file_list = os.listdir(log_params["source"])
file_list.sort()
file_list.sort(key=len)

model_params = config['model_params']
print(f'Source directory: {log_params["source"]}')

for check_path in file_list:
    path = log_params["source"] + '/' + check_path

    checkpoint = torch.load(path)
    print()
    print(f"Epoch: {checkpoint['epoch']}")

    epochs.append(checkpoint['epoch'])
    tr_loss.append(checkpoint['tr_loss'])

    model_params['checkpoint'] = path
    v_l = lf.validation(val_loader, model_params)
    val_loss.append(v_l)

csv_dict = {'EPOCH': epochs, 'TRAINING LOSS': tr_loss, 'VALIDATION LOSS': val_loss}
csv_df = pandas.DataFrame(csv_dict)
csv_df.to_csv(log_params['dest'], index=False)
print(f'Csv file saved: {log_params["dest"]}')
