import os

class PureSentencePieceTokenizer:
    """
    Pure Python SentencePiece-compatible tokenizer that reads Query_Tokenizer.vocab.
    Works on all platforms and Python versions without C-extension DLL issues.
    """
    SPIECE_UNDERLINE = '\u2581'

    def __init__(self, vocab_file="Query_Tokenizer.vocab"):
        self.vocab = {}
        self.id_to_token = {}
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 0
        self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        if not os.path.exists(vocab_file):
            return

        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                token = parts[0]
                score = float(parts[1]) if len(parts) > 1 else 0.0
                self.vocab[token] = {"id": idx, "score": score}
                self.id_to_token[idx] = token

        if "<s>" in self.vocab:
            self.bos_id = self.vocab["<s>"]["id"]
        if "</s>" in self.vocab:
            self.eos_id = self.vocab["</s>"]["id"]
        if "<unk>" in self.vocab:
            self.unk_id = self.vocab["<unk>"]["id"]

    def encode(self, text: str, out_type=int):
        """
        Tokenizes input text matching the SentencePiece BPE vocabulary.
        """
        norm_text = text.replace(" ", self.SPIECE_UNDERLINE)
        if not norm_text.startswith(self.SPIECE_UNDERLINE):
            norm_text = self.SPIECE_UNDERLINE + norm_text

        tokens = []
        token_ids = []
        i = 0
        n = len(norm_text)
        sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)

        while i < n:
            matched = False
            for piece in sorted_tokens:
                if piece in ["<s>", "</s>", "<unk>"]:
                    continue
                if norm_text.startswith(piece, i):
                    tokens.append(piece)
                    token_ids.append(self.vocab[piece]["id"])
                    i += len(piece)
                    matched = True
                    break
            if not matched:
                char = norm_text[i]
                if char in self.vocab:
                    tokens.append(char)
                    token_ids.append(self.vocab[char]["id"])
                else:
                    tokens.append(char)
                    token_ids.append(self.unk_id)
                i += 1

        if out_type == str:
            return tokens
        return token_ids

    def decode(self, token_ids):
        pieces = [self.id_to_token.get(tid, "") for tid in token_ids]
        raw = "".join(pieces)
        return raw.replace(self.SPIECE_UNDERLINE, " ").strip()

    def id_to_piece(self, tid):
        piece = self.id_to_token.get(tid, "<unk>")
        return piece.replace(self.SPIECE_UNDERLINE, "_")

    def piece_to_id(self, piece):
        return self.vocab.get(piece, {}).get("id", self.unk_id)

    def GetPieceSize(self):
        return len(self.vocab)


def get_tokenizer(base_dir="."):
    """
    Attempts to load native sentencepiece if available, otherwise falls back to pure Python.
    """
    model_path = os.path.join(base_dir, "Query_Tokenizer.model")
    vocab_path = os.path.join(base_dir, "Query_Tokenizer.vocab")

    # On Python 3.14+, C-extension wheels can segfault; use pure-Python tokenizer
    import sys
    if sys.version_info < (3, 14):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            if os.path.exists(model_path):
                sp.Load(model_path)
                return sp, True
        except Exception:
            pass

    # Pure Python universal tokenizer
    if os.path.exists(vocab_path):
        tok = PureSentencePieceTokenizer(vocab_path)
        return tok, True

    return None, False
