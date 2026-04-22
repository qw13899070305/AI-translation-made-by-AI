# enhanced_data_loader.py —— 多源数据自动下载与导出
import os
import sys
from datasets import load_dataset, concatenate_datasets

# 数据源配置（按需增删）
DATA_SOURCES = [
    {
        "name": "wikipedia",
        "subset": "20220301.zh",
        "split": "train",
        "desc": "维基百科中文",
        "max_samples": 30000,
    },
    {
        "name": "wikitext",
        "subset": "wikitext-103-v1",
        "split": "train",
        "desc": "英文维基文本",
        "max_samples": 30000,
    },
    {
        "name": "codeparrot/codeparrot",
        "subset": None,
        "split": "train",
        "desc": "代码数据",
        "max_samples": 20000,
    },
    {
        "name": "Open-Orca/OpenOrca",
        "subset": None,
        "split": "train",
        "desc": "指令对话",
        "max_samples": 30000,
    },
    {
        "name": "GAIR/MegaScience",
        "subset": None,
        "split": "train",
        "desc": "科学推理",
        "max_samples": 20000,
    },
]

def load_diverse_datasets(output_file="enhanced_data.txt"):
    """加载多源数据集并导出为纯文本文件"""
    all_texts = []

    for src in DATA_SOURCES:
        try:
            print(f"📥 加载 {src['desc']} ({src['name']})...")
            if src["subset"]:
                ds = load_dataset(src["name"], src["subset"], split=src["split"], streaming=False)
            else:
                ds = load_dataset(src["name"], split=src["split"], streaming=False)
            
            # 限制样本数量
            if len(ds) > src["max_samples"]:
                ds = ds.select(range(src["max_samples"]))
            
            # 提取文本字段
            for sample in ds:
                text = None
                # 根据不同数据集的字段名提取
                if "text" in sample:
                    text = sample["text"]
                elif "content" in sample:
                    text = sample["content"]
                elif "question" in sample and "response" in sample:
                    text = f"问：{sample['question']} 答：{sample['response']}"
                elif "instruction" in sample and "output" in sample:
                    text = f"问：{sample['instruction']} 答：{sample['output']}"
                else:
                    continue
                
                if text:
                    # 去除换行符，每行一个样本
                    text = text.replace("\n", " ").strip()
                    if text:
                        all_texts.append(text)
            
            print(f"   ✅ 提取 {len(all_texts)} 条累计")
        except Exception as e:
            print(f"   ⚠️ 加载失败: {e}")

    if not all_texts:
        print("❌ 没有提取到任何文本，请检查网络或数据集")
        sys.exit(1)

    with open(output_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")

    print(f"\n🎉 增强数据已保存到 {output_file}，共 {len(all_texts)} 条")
    print("📌 现在你可以在 config.py 的 text_datasets 中添加 'enhanced_data.txt' 来使用这些数据。")

if __name__ == "__main__":
    load_diverse_datasets()