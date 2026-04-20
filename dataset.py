import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import sentencepiece as spm
from config import Config
from utils import format_chat_prompt

cfg = Config()

class TextChatDataset(Dataset):
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
        self.data = []
        print("加载文本数据集...")
        all_splits = []
        for ds_name in cfg.text_datasets:
            ds = load_dataset(ds_name, split="train", streaming=False)
            ds = ds.select(range(min(cfg.max_samples_per_dataset, len(ds))))
            all_splits.append(ds)
        combined = concatenate_datasets(all_splits) if len(all_splits) > 1 else all_splits[0]
        for sample in combined:
            q = sample.get('question') or sample.get('instruction')
            a = sample.get('response') or sample.get('output')
            if q and a:
                text = format_chat_prompt(q, a)
                ids = self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)
                if len(ids) > cfg.max_seq_len:
                    ids = ids[:cfg.max_seq_len-1] + [self.sp.eos_id()]
                self.data.append(torch.tensor(ids, dtype=torch.long))
        print(f"文本数据共 {len(self.data)} 条")

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)