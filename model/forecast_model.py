import torch
from torch import nn

from model.decoders.attention_decoder import AttentionDecoder
from model.encoders.city_encoder import CityEncoder


class ForecastModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=1, dropout=False):
        super().__init__()

        self.hs = hidden_size

        self.encoder = CityEncoder(input_size=input_size, output_size=hidden_size, hidden_layers=hidden_layers, dropout=dropout)
        self.decoder = AttentionDecoder(input_size=hidden_size, output_size=output_size, hidden_layers=hidden_layers, dropout=dropout)

    def forward(self, geo, weather, target_idx, mask):
        B, N, _, _ = weather.shape
        embeddings = []

        for i in range(N):
            if mask[0, i] == 0:
                embeddings.append(torch.zeros(B, self.hs))
                continue
            emb = self.encoder(geo[:, i], weather[:, i])
            embeddings.append(emb)

        city_embeds = torch.stack(embeddings, dim=1)
        return self.decoder(city_embeds, target_idx, mask)
