import os
import sentencepiece as spm
from datasets import load_dataset
from config import Config
from utils import ensure_dir

cfg = Config()
ensure_dir("./tokenizer")

print("Downloading dataset for tokenizer training...")
dataset = load_dataset(cfg.text_datasets[0], split="train", streaming=True)
texts = []
for i, sample in enumerate(dataset):
    if i >= 2000:
        break
    q = sample.get('question') or sample.get('instruction')
    a = sample.get('response') or sample.get('output')
    if q and a:
        texts.append(f"### User: {q}\n### Assistant: {a}")

with open("tokenizer_train_text.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")

print("Training BPE tokenizer...")
spm.SentencePieceTrainer.train(
    input="tokenizer_train_text.txt",
    model_prefix=f"tokenizer/{cfg.tokenizer_prefix}",
    vocab_size=cfg.vocab_size,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]"
)
print(f"Tokenizer saved to tokenizer/{cfg.tokenizer_prefix}.model")
os.remove("tokenizer_train_text.txt")