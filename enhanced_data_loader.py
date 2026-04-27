# enhanced_data_loader.py —— 多源数据自动下载与导出（追加模式）
import os
import sys
from datasets import load_dataset

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
    """加载多源数据集并追加到纯文本文件中"""
    # 先读取文件中已有的内容，用于去重
    existing_texts = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                existing_texts.add(line.strip())

    new_count = 0

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
            
            # 准备一批新文本，准备追加写入
            new_texts = []
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
                    text = text.replace("\n", " ").strip()
                    # 去重：只添加文件中不存在的新内容
                    if text and text not in existing_texts:
                        new_texts.append(text)
                        existing_texts.add(text)  # 加入集合，防止本次加载的内部重复
            
            # 追加写入文件
            if new_texts:
                with open(output_file, "a", encoding="utf-8") as f:
                    for text in new_texts:
                        f.write(text + "\n")
                new_count += len(new_texts)
                print(f"   ✅ 新增 {len(new_texts)} 条，累计 {len(existing_texts)} 条")
            else:
                print(f"   ⏭️ 无新内容，跳过")
                
        except Exception as e:
            print(f"   ⚠️ 加载失败: {e}")

    if new_count == 0:
        print("❌ 没有提取到任何新文本，文件保持不变")
    else:
        print(f"\n🎉 成功追加 {new_count} 条新数据到 {output_file}")
        print("📌 config.py 的 text_datasets 中已包含 'enhanced_data.txt'，可直接训练。")

if __name__ == "__main__":
    load_diverse_datasets()