import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
import sentencepiece as spm
from config import Config
from utils import format_chat_prompt
import os

cfg = Config()

class TextChatDataset(Dataset):
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        tokenizer_path = f"tokenizer/{cfg.tokenizer_prefix}.model"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}. Please run tokenizer_train.py first.")
        self.sp.load(tokenizer_path)
        self.data = []
        print("Loading text datasets...")
        all_splits = []

        for ds_name in cfg.text_datasets:
            try:
                if ds_name.endswith('.txt'):
                    if not os.path.exists(ds_name):
                        print(f"   File not found, skipping: {ds_name}")
                        continue
                    print(f"   Loading local file: {ds_name}")
                    with open(ds_name, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    ds = HFDataset.from_dict({'text': lines})
                else:
                    print(f"   Loading remote dataset: {ds_name}")
                    ds = load_dataset(ds_name, split="train", streaming=False)
                    ds = ds.select(range(min(cfg.max_samples_per_dataset, len(ds))))
                all_splits.append(ds)
                print(f"   Loaded {len(ds)} entries")
            except Exception as e:
                print(f"   Failed to load {ds_name}: {e}")

        if not all_splits:
            raise RuntimeError("No datasets loaded successfully")

        combined = concatenate_datasets(all_splits) if len(all_splits) > 1 else all_splits[0]

        processed = 0
        for sample in combined:
            q = a = None
            if 'question' in sample and 'response' in sample:
                q, a = sample['question'], sample['response']
            elif 'instruction' in sample and 'output' in sample:
                q, a = sample['instruction'], sample['output']
            elif 'text' in sample:
                text = sample['text']
                if len(text) > 50:
                    q, a = text[:len(text)//2], text[len(text)//2:]
            if not q or not a:
                continue

            text = format_chat_prompt(str(q).strip(), str(a).strip())
            ids = self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)
            if len(ids) > cfg.max_seq_len:
                ids = ids[:cfg.max_seq_len-1] + [self.sp.eos_id()]
            self.data.append(torch.tensor(ids, dtype=torch.long))
            processed += 1

        print(f"Valid entries: {processed}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    sp = spm.SentencePieceProcessor()
    sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), sp.pad_id(), dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded[:, :-1], padded[:, 1:]


def get_dataloader(batch_size=cfg.batch_size):
    dataset = TextChatDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True if cfg.device == "cuda" else False, persistent_workers=True if cfg.num_workers > 0 else False)