import torch
import torch.nn as nn
import torch.nn.functional as func


class AttentionDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, horizon_size=24, hidden_layers=1, dropout=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.horizon = horizon_size
        self.output_size = output_size

        self.hl = hidden_layers
        self.hidden_layers = nn.ModuleList()

        self.dr = dropout
        self.dropout_layers = nn.ModuleList()

        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        # hidden layers
        if self.hl > 0:
            for _ in range(self.hl):
                self.hidden_layers.append(nn.Linear(hidden_size+2, hidden_size+2))
                if self.dr:
                    self.dropout_layers.append(nn.Dropout(0.3))

        self.output_layer = nn.Linear(hidden_size+2, output_size)

    def forward(self, embeddings, target_idx, mask):
        B, N, D = embeddings.shape
        q = embeddings[torch.arange(B), target_idx]
        Q = self.query(q).unsqueeze(1)
        K = self.key(embeddings)
        V = self.value(embeddings)

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (D ** 0.5)

        mask = mask.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V).squeeze(1)
        attended = attended.unsqueeze(1).expand(-1, self.horizon, -1)

        # cyclic time encoding for each horizon step
        t = torch.arange(self.horizon).float()
        t = t / self.horizon
        sin_t = torch.sin(2 * torch.pi * t)
        cos_t = torch.cos(2 * torch.pi * t)
        time_encoding = torch.stack([sin_t, cos_t], dim=1)
        time_encoding = time_encoding.unsqueeze(0).expand(B, -1, -1)

        hidden_input = torch.cat([attended, time_encoding], dim=-1)

        if self.hl > 0:
            for i, layer in enumerate(self.hidden_layers):
                hidden_input = func.relu(layer(hidden_input))
                if self.dr:
                    hidden_input = self.dropout_layers[i](hidden_input)

        output = self.output_layer(hidden_input)
        return output
