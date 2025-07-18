import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class WeatherNetwork(nn.Module):
    def __init__(self, input_size, output_size, horizon_size=24, hidden_size=64, hidden_layers=0, dropout=False):
        super().__init__()
        self.hs = horizon_size

        self.hl = hidden_layers
        self.hidden_layers = nn.ModuleList()

        self.dr = dropout
        self.dropout_layers = nn.ModuleList()

        # for geographic info
        self.geo_layer = nn.Linear(3, 16)

        # for other weather info
        self.lstm = nn.LSTM(input_size=input_size+16, hidden_size=hidden_size, batch_first=True)

        # hidden layers
        if self.hl > 0:
            for _ in range(self.hl):
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
                if self.dr:
                    self.dropout_layers.append(nn.Dropout(0.3))

        self.output_layer = nn.Linear(hidden_size+2, output_size)

    def forward(self, geo_data, weather_data):
        geo_emb = func.relu(self.geo_layer(geo_data))
        geo_emb_seq = geo_emb.unsqueeze(1).expand(-1, weather_data.size(1), -1)

        lstm_input = torch.cat([weather_data, geo_emb_seq], dim=-1)

        lstm_emb, _ = self.lstm(lstm_input)
        features = lstm_emb[:, -1, :]

        if self.hl > 0:
            for i, layer in enumerate(self.hidden_layers):
                features = func.relu(layer(features))
                if self.dr:
                    features = self.dropout_layers[i](features)

        features_seq = features.unsqueeze(1).repeat(1, self.hs, 1)

        batch_size = features.size(0)
        device = features.device

        t = torch.arange(self.hs, device=device).float() / self.hs

        sin_t = torch.sin(2 * np.pi * t)
        cos_t = torch.cos(2 * np.pi * t)

        time_encoding = torch.stack([sin_t, cos_t], dim=1)

        time_encoding = time_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        decoder_input = torch.cat([features_seq, time_encoding], dim=-1)

        output = self.output_layer(decoder_input)

        return output
