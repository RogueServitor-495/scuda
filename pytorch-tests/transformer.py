import torch
import torch.nn as nn
import time

# 超参数（都设得比较小以节省显存）
BATCH_SIZE = 2
SEQ_LEN     = 32
VOCAB_SIZE  = 1000
EMB_DIM     = 64
FF_DIM      = 256
NUM_HEADS   = 4
NUM_LAYERS  = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

# 位置编码（sinusoidal）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))  # (max_len,1,d_model)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)]

# 主模型
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, ff_dim, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_enc   = PositionalEncoding(emb_dim, max_len=SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,            # 推理时关闭 dropout
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, src):
        """
        src: (seq_len, batch)  长度为 SEQ_LEN 的 token id 序列
        """
        # token embedding + positional encoding
        x = self.token_emb(src)              # (seq_len, batch, emb_dim)
        x = self.pos_enc(x)                  # (seq_len, batch, emb_dim)
        x = self.encoder(x)                  # (seq_len, batch, emb_dim)
        logits = self.decoder(x)             # (seq_len, batch, vocab_size)
        return logits

# 实例化并搬到 GPU
start = time.time()
model = SmallTransformer(VOCAB_SIZE, EMB_DIM, FF_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
model.eval()

# 假输入
src = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE), device=DEVICE)

# 推理，开启半精度
with torch.no_grad():
    with torch.cuda.amp.autocast():
        logits = model(src)  # (seq_len, batch, vocab_size)
        # 取最后一步最可能的 token
        next_tokens = logits.argmax(dim=-1)[-1]  # (batch,)

print("Next tokens:", next_tokens)
print("Peak memory:", torch.cuda.max_memory_allocated() / 1024**2, "MiB")
end = time.time()

print("time cost:", end-start)
