from datasets import load_dataset
from config import Config

cfg = Config()
print("📂 Current datasets:")
for ds_name in cfg.text_datasets:
    print(f" - {ds_name}")

print("\n🔍 Previewing first 3 samples from first dataset:")
try:
    dataset = load_dataset(cfg.text_datasets[0], split="train", streaming=True)
    for i, sample in enumerate(dataset):
        if i >= 3: break
        print(f"--- Sample {i+1} ---")
        for k, v in sample.items():
            print(f"{k}: {str(v)[:100]}...")
except Exception as e:
    print(f"Preview failed: {e}")