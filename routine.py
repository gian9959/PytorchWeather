import json
from torch.utils.data import DataLoader

import learning_functions as lf
import utils
import model.weather_dataset as mw

tr_files = ["2000-01-01-2009-12-31", "2010-01-01-2019-12-31"]
val_files = ["2020-01-01-2024-12-31"]

with open('config.json', 'r') as f:
    config = json.load(f)

print('Loading training dataset...')
tr_dataset = utils.load_all(tr_files)
tr_loader = DataLoader(tr_dataset, batch_size=32, collate_fn=mw.collate_fn, shuffle=True)

# print('Loading validation dataset...')
# val_dataset = utils.load_all(val_files)
# val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=mw.collate_fn, shuffle=False)

model_params = config['model_params']

for i in range(10):
    model_params['checkpoint'] = lf.training(tr_loader, model_params)
    # lf.validation(val_loader, model_params)

