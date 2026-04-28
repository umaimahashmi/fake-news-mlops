import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class LightweightFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
        use_attention=True,
        use_ffn=True,
        use_residual=True,
        use_layernorm=True,
        use_dropout=True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_ffn = use_ffn
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        d = dropout if use_dropout else 0.0
        self.attn = EnhancedMultiHeadAttention(d_model, num_heads, d)
        self.ffn = LightweightFFN(d_model, d_ff, d)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(d)
        self.drop2 = nn.Dropout(d)

    def forward(self, x, mask=None):
        if self.use_attention:
            attn_out = self.drop1(self.attn(x, mask))
            x = x + attn_out if self.use_residual else attn_out
            x = self.norm1(x) if self.use_layernorm else x
        if self.use_ffn:
            ffn_out = self.drop2(self.ffn(x))
            x = x + ffn_out if self.use_residual else ffn_out
            x = self.norm2(x) if self.use_layernorm else x
        return x


class OptimalFNDModel(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        d_model=192,
        num_heads=6,
        num_layers=4,
        d_ff=512,
        max_len=128,
        dropout=0.1,
        num_classes=2,
        use_positional_encoding=True,
        use_attention=True,
        use_ffn=True,
        use_residual=True,
        use_layernorm=True,
        use_dropout=True,
        use_cls_token=True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.use_positional_encoding = use_positional_encoding
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embed_scale = math.sqrt(d_model)
        self.pos_enc = PositionalEncoding(
            d_model, max_len, dropout if use_dropout else 0.0
        )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    use_attention=use_attention,
                    use_ffn=use_ffn,
                    use_residual=use_residual,
                    use_layernorm=use_layernorm,
                    use_dropout=use_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model) if use_layernorm else nn.Identity(),
            nn.Dropout(dropout if use_dropout else 0.0),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) * self.embed_scale
        if self.use_positional_encoding:
            x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        x = self.final_norm(x)
        if self.use_cls_token:
            cls_output = x[:, 0, :]
        else:
            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(-1).float()
                cls_output = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            else:
                cls_output = x.mean(dim=1)
        return self.classifier(cls_output)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_CONFIGS = {
    "bert": dict(d_model=192, num_heads=6, num_layers=4, d_ff=512),
    "roberta": dict(d_model=128, num_heads=4, num_layers=3, d_ff=384),
    "deberta": dict(d_model=128, num_heads=4, num_layers=3, d_ff=384),
    "distilbert": dict(
        d_model=128, num_heads=4, num_layers=3, d_ff=384, use_cls_token=False
    ),
}
