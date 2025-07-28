import os

import torch
from alive_progress import alive_bar
from torch import nn

from model.forecast_model import ForecastModel


def training(loader, model_params):
    starting_epoch = 1
    train_length = model_params['train_length']
    hidden_layers = model_params['hidden_layers']
    dropout = model_params['dropout']

    dr_string = ''

    if os.path.isfile(model_params['checkpoint']):
        print(f'Loading checkpoint: {model_params["checkpoint"]}')
        checkpoint = torch.load(model_params['checkpoint'])

        if checkpoint['epoch'] is not None:
            starting_epoch = checkpoint['epoch'] + 1

        if checkpoint['hidden_layers'] is not None:
            hidden_layers = checkpoint['hidden_layers']

        if checkpoint['dropout'] is not None:
            dropout = checkpoint['dropout']

        model = ForecastModel(input_size=11, output_size=6, hidden_layers=hidden_layers, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Checkpoint NOT loaded")
        model = ForecastModel(input_size=11, output_size=6, hidden_layers=hidden_layers, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Hidden layers: {hidden_layers}")

    if dropout:
        dr_string = '_dr'
    print(f"Dropout: {dropout}")

    print('Training...')
    loss_fn = nn.SmoothL1Loss()
    model.train()

    for epoch in range(starting_epoch, starting_epoch + train_length):
        tr_loss = 0.0
        with alive_bar(len(loader)) as bar:
            for inputs, labels, geos, mask, targets in loader:
                scores = model(geos, inputs, targets, mask)
                loss = loss_fn(scores, labels[torch.arange(labels.size(0)), targets])
                tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar()

        tr_loss = tr_loss / len(loader)
        print(f"Epoch {epoch}: Loss {tr_loss:.4f}")

        checkpoint = {'epoch': epoch, 'hidden_layers': hidden_layers, 'dropout': dropout,
                      'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'tr_loss': tr_loss}

        save_path = f"../checkpoints/{hidden_layers}H{dr_string}/Epoch{epoch}_checkpoint.pth"
        torch.save(checkpoint, save_path)

    return save_path


def validation(loader, model_params):
    hidden_layers = model_params['hidden_layers']
    dropout = model_params['dropout']

    checkpoint = torch.load(model_params['checkpoint'])

    if checkpoint['hidden_layers'] is not None:
        hidden_layers = checkpoint['hidden_layers']

    if checkpoint['dropout'] is not None:
        dropout = checkpoint['dropout']

    model = ForecastModel(input_size=11, output_size=6, hidden_layers=hidden_layers, dropout=dropout)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    loss_fn = nn.SmoothL1Loss()
    val_loss = 0.0

    with torch.no_grad():
        with alive_bar(len(loader)) as bar:
            for inputs, labels, geos, mask, targets in loader:
                scores = model(geos, inputs, targets, mask)
                loss = loss_fn(scores, labels[torch.arange(labels.size(0)), targets])
                val_loss += loss.item()
                bar()

    print(f"Validation Loss: {val_loss / len(loader):.4f}")

    return val_loss / len(loader)
