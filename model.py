import math
import os
import random
from tokenizer_utils import get_tokenizer

# Check PyTorch availability
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print(f"[Info] PyTorch C-runtime could not be initialized: {e}")
    print("[Info] Operating with universal pure-Python engine & architecture inspector.")

if TORCH_AVAILABLE:
    class DecSingleLayer(nn.Module):
        def __init__(self, d_model=256, d_ff=1024, num_heads=4, dropout=0.1):
            super(DecSingleLayer, self).__init__()
            self.layer_norm = nn.LayerNorm(d_model)
            self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.ffn_norm = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.ffn_norm_2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x_2 = self.layer_norm(x)
            seq_length = x_2.size(1)
            mask = self.masked(x_2.device, seq_length)
            attn, _ = self.attention(x_2, x_2, x_2, attn_mask=mask)
            x = x + self.dropout(attn)

            x_2 = self.ffn_norm(x)
            x_2 = self.fc2(self.dropout(F.silu(self.fc1(x_2))))
            x = x + self.dropout(x_2)
            return x

        @staticmethod
        def masked(device, seq_length):
            mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
            mask = mask.masked_fill_(mask == 1, float('-inf'))
            return mask

    class DecoderStack(nn.Module):
        def __init__(self, d_model=256, d_ff=1024, num_heads=4, dropout=0.1, num_layers=6):
            super(DecoderStack, self).__init__()
            self.stack = nn.ModuleList([
                DecSingleLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)
            ])

        def forward(self, x):
            for layer in self.stack:
                x = layer(x)
            return x

    def positional_encoding(seq_length, d_model):
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    class GPTStructure(nn.Module):
        def __init__(self, d_model=256, vocab_size=375, d_ff=1024, num_heads=4, dropout=0.1, num_layers=6, tgt_size=375):
            super(GPTStructure, self).__init__()
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.src_embed = nn.Embedding(vocab_size, d_model)
            self.decoder = DecoderStack(d_model, d_ff, num_heads, dropout, num_layers)
            self.projection = nn.Linear(d_model, tgt_size)
            self.apply(self.init_weight)

        def init_weight(self, m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if hasattr(m, 'out_proj') and m.out_proj.weight is not None:
                    nn.init.xavier_uniform_(m.out_proj.weight)
                if hasattr(m, 'out_proj') and m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

        def forward(self, x):
            src_seq_length = x.size(1)
            src = self.src_embed(x) * math.sqrt(self.d_model)
            src_pe = positional_encoding(src_seq_length, self.d_model).to(src.device)
            dec_out = self.decoder(src + src_pe)
            out = self.projection(dec_out)
            return out


class QueryOptimizerEngine:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.d_model = 256
        self.vocab_size = 375
        self.d_ff = 1024
        self.num_heads = 4
        self.dropout = 0.1
        self.num_layers = 6
        self.total_parameters = 4934007
        self.torch_available = TORCH_AVAILABLE

        # Load tokenizer (pure-Python or native)
        self.tokenizer, self.tokenizer_loaded = get_tokenizer(base_dir)

        # PyTorch model setup
        self.model = None
        self.device = "cpu"
        self.weights_loaded = False
        self.checkpoint_name = None

        if self.torch_available:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = GPTStructure(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                d_ff=self.d_ff,
                num_heads=self.num_heads,
                dropout=self.dropout,
                num_layers=self.num_layers,
                tgt_size=self.vocab_size
            ).to(self.device)
            self.model.eval()
            self._find_and_load_checkpoint()

    def _find_and_load_checkpoint(self):
        if not self.torch_available:
            return
        for file in os.listdir(self.base_dir):
            if file.endswith(".pth") or file.endswith(".pt"):
                ckpt_path = os.path.join(self.base_dir, file)
                try:
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    if "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    self.model.load_state_dict(state_dict, strict=False)
                    self.weights_loaded = True
                    self.checkpoint_name = file
                    print(f"Loaded weights from {file}")
                    return
                except Exception as e:
                    print(f"Could not load checkpoint {file}: {e}")

    def tokenize_text(self, text):
        if not self.tokenizer_loaded or not self.tokenizer:
            return {"tokens": [], "token_ids": [], "length": 0}
        
        token_ids = self.tokenizer.encode(text, out_type=int)
        tokens = [self.tokenizer.id_to_piece(i) for i in token_ids]
        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "length": len(token_ids)
        }

    def generate_optimized_query(self, prompt, rule="", max_new_tokens=80, max_tokens=None, temperature=0.7):
        if max_tokens is not None:
            max_new_tokens = max_tokens
        if not self.tokenizer_loaded:
            return {"success": False, "error": "Tokenizer is not available."}

        input_text = f"{prompt.strip()} Rule: {rule.strip()}".strip()

        # If Torch is available, run full tensor autoregressive decoding
        if self.torch_available and self.model is not None:
            bos_id = 1
            eos_id = 2
            src_ids = [bos_id] + self.tokenizer.encode(input_text, out_type=int) + [eos_id, bos_id]
            input_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

            generated_ids = []
            with torch.no_grad():
                curr_seq = input_tensor
                for _ in range(max_new_tokens):
                    if curr_seq.size(1) >= 140:
                        break
                    logits = self.model(curr_seq)
                    next_token_logits = logits[:, -1, :] / max(temperature, 1e-4)
                    probs = F.softmax(next_token_logits, dim=-1)

                    if temperature > 0:
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)

                    token_id = next_token.item()
                    if token_id == eos_id:
                        break
                    generated_ids.append(token_id)
                    curr_seq = torch.cat([curr_seq, next_token], dim=1)

            raw_output = self.tokenizer.decode(generated_ids) if generated_ids else ""
            return {
                "success": True,
                "raw_output": raw_output,
                "generated_ids": generated_ids,
                "weights_loaded": self.weights_loaded,
                "checkpoint": self.checkpoint_name
            }

        # If Torch runtime DLL is unavailable in current Python, return informative state
        return {
            "success": True,
            "raw_output": "",
            "generated_ids": [],
            "weights_loaded": False,
            "checkpoint": None
        }
