from datasets import load_dataset
from config import Config

cfg = Config()
print("📂 当前配置的数据集：")
for ds_name in cfg.text_datasets:
    print(f" - {ds_name}")

print("\n🔍 预览第一个数据集的前 3 条样本：")
dataset = load_dataset(cfg.text_datasets[0], split="train", streaming=True)
for i, sample in enumerate(dataset):
    if i >= 3:
        break
    print(f"--- Sample {i+1} ---")
    for k, v in sample.items():
        print(f"{k}: {str(v)[:100]}...")