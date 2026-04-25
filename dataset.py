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
            raise FileNotFoundError(f"分词器不存在: {tokenizer_path}\n请先运行: python tokenizer_train.py")
        self.sp.load(tokenizer_path)
        self.data = []
        print("📚 加载文本数据集...")
        all_splits = []

        for ds_name in cfg.text_datasets:
            try:
                if ds_name.endswith('.txt'):
                    # 本地文本文件
                    if not os.path.exists(ds_name):
                        print(f"   ⚠️ 文件不存在，跳过: {ds_name}")
                        continue
                    print(f"   📄 加载本地文件: {ds_name}")
                    with open(ds_name, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    ds = HFDataset.from_dict({'text': lines})
                else:
                    # HuggingFace 远程数据集
                    print(f"   📡 加载远程数据集: {ds_name}")
                    ds = load_dataset(ds_name, split="train", streaming=False)
                    ds = ds.select(range(min(cfg.max_samples_per_dataset, len(ds))))
                all_splits.append(ds)
                print(f"   ✅ 加载 {len(ds)} 条")
            except Exception as e:
                print(f"   ❌ 加载失败 {ds_name}: {e}")

        if not all_splits:
            raise RuntimeError("没有成功加载任何数据集")

        combined = concatenate_datasets(all_splits) if len(all_splits) > 1 else all_splits[0]

        processed = 0
        for sample in combined:
            # 灵活的字段提取
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

        print(f"✅ 有效数据共 {processed} 条")

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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device == "cuda" else False,
        persistent_workers=True if cfg.num_workers > 0 else False
    )