import torch
import torch.nn as nn
import torch.nn.functional as func


class AttentionDecoder(nn.Module):
    def __init__(self, input_size, output_size, horizon_size=24, hidden_layers=1, dropout=False):
        super().__init__()

        self.hs = horizon_size
        self.output_size = output_size

        self.hl = hidden_layers
        self.hidden_layers = nn.ModuleList()

        self.dr = dropout
        self.dropout_layers = nn.ModuleList()

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        # hidden layers
        if self.hl > 0:
            for _ in range(self.hl):
                self.hidden_layers.append(nn.Linear(input_size+2, input_size+2))
                if self.dr:
                    self.dropout_layers.append(nn.Dropout(0.3))

        self.output_layer = nn.Linear(input_size+2, output_size)

    def forward(self, embeddings, target_idx):
        B, N, D = embeddings.shape
        q = embeddings[torch.arange(B), target_idx]
        Q = self.query(q).unsqueeze(1)
        K = self.key(embeddings)
        V = self.value(embeddings)

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (D ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V).squeeze(1)
        attended = attended.unsqueeze(1).expand(-1, self.hs, -1)

        # cyclic time encoding for each horizon step
        device = embeddings.device
        t = torch.arange(self.hs, device=device).float()
        t = t / self.hs
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
