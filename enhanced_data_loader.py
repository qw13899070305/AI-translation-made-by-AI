import os, sys, glob
from datasets import load_dataset, Dataset as HFDataset

DATA_SOURCES = [
    {"name": "wikipedia", "subset": "20220301.zh", "split": "train", "desc": "维基百科中文", "max_samples": 30000},
    {"name": "wikitext", "subset": "wikitext-103-v1", "split": "train", "desc": "英文维基文本", "max_samples": 30000},
    {"name": "codeparrot/codeparrot", "subset": None, "split": "train", "desc": "代码数据", "max_samples": 20000},
    {"name": "Open-Orca/OpenOrca", "subset": None, "split": "train", "desc": "指令对话", "max_samples": 30000},
    {"name": "GAIR/MegaScience", "subset": None, "split": "train", "desc": "科学推理", "max_samples": 20000},
]

def load_diverse_datasets(output_file="enhanced_data.txt"):
    existing_texts = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f: existing_texts.add(line.strip())

    total_new = 0
    for src in DATA_SOURCES:
        try:
            print(f"📥 Loading {src['desc']} ({src['name']})...")
            if src.get("subset"): ds = load_dataset(src["name"], src["subset"], split=src["split"], streaming=False)
            else: ds = load_dataset(src["name"], split=src["split"], streaming=False)
            if len(ds) > src["max_samples"]: ds = ds.select(range(src["max_samples"]))
            new_texts = []
            for sample in ds:
                text = sample.get("text") or sample.get("content") or sample.get("sentence")
                if not text:
                    for v in sample.values():
                        if isinstance(v, str) and len(v) > 10: text = v; break
                if text: text = text.replace("\n", " ").strip()
                if text and text not in existing_texts: new_texts.append(text); existing_texts.add(text)
            if new_texts:
                with open(output_file, "a", encoding="utf-8") as f:
                    for t in new_texts: f.write(t + "\n")
                total_new += len(new_texts)
                print(f"   ✅ Added {len(new_texts)}, total {len(existing_texts)}")
            else: print(f"   ⏭️  No new content")
        except Exception as e: print(f"   ⚠️ Load failed: {e}")
    if total_new == 0: print("❌ No new data extracted")
    else: print(f"\n🎉 Appended {total_new} entries to {output_file}")

if __name__ == "__main__": load_diverse_datasets()