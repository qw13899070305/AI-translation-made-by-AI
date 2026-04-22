# multitask_trainer.py —— 多任务学习训练器
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from config import Config
from model import MiniChat
from lora import apply_lora_to_model, mark_only_lora_as_trainable
import sentencepiece as spm

cfg = Config()
device = cfg.device

# 加载分词器
sp = spm.SentencePieceProcessor()
sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
vocab_size = sp.get_piece_size()

class MultiTaskTrainer:
    def __init__(self):
        self.model = MiniChat(vocab_size).to(device)
        apply_lora_to_model(self.model)
        mark_only_lora_as_trainable(self.model)
        
        # 任务权重（可调整）
        self.task_weights = {
            "lm": 0.5,        # 语言建模
            "qa": 0.3,        # 问答
            "sentiment": 0.2  # 情感分析
        }
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate
        )
    
    def prepare_sample(self, text, task_type="lm"):
        """将文本转换为模型输入格式"""
        # 根据任务类型添加不同前缀
        if task_type == "qa":
            text = f"问答：{text}"
        elif task_type == "sentiment":
            text = f"情感分析：{text}"
        
        ids = sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        if len(ids) > cfg.max_seq_len:
            ids = ids[:cfg.max_seq_len-1] + [sp.eos_id()]
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        targets = input_ids.clone()
        return input_ids, targets
    
    def train_step(self, batch_texts, task_type):
        """单步训练"""
        self.model.train()
        total_loss = 0
        
        for text in batch_texts:
            input_ids, targets = self.prepare_sample(text, task_type)
            _, loss, _ = self.model(input_ids, targets=targets)
            weighted_loss = self.task_weights[task_type] * loss
            
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()
            total_loss += weighted_loss.item()
        
        return total_loss / len(batch_texts)
    
    def train(self, lm_data, qa_data, sentiment_data, epochs=5):
        """完整训练流程"""
        print("🚀 开始多任务学习...")
        print(f"任务权重: {self.task_weights}")
        print(f"数据量 - LM: {len(lm_data)}, QA: {len(qa_data)}, Sentiment: {len(sentiment_data)}")
        
        for epoch in range(1, epochs+1):
            # 混合采样训练（简化版，实际可更复杂）
            import random
            all_tasks = []
            all_tasks.extend([("lm", t) for t in lm_data])
            all_tasks.extend([("qa", t) for t in qa_data])
            all_tasks.extend([("sentiment", t) for t in sentiment_data])
            random.shuffle(all_tasks)
            
            batch_size = 8
            total_loss = 0
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i:i+batch_size]
                batch_texts = [item[1] for item in batch]
                task_type = batch[0][0]  # 取第一个任务类型（简化）
                loss = self.train_step(batch_texts, task_type)
                total_loss += loss
            
            avg_loss = total_loss / (len(all_tasks) // batch_size)
            print(f"Epoch {epoch}, 平均损失: {avg_loss:.4f}")
        
        # 保存模型
        torch.save(self.model.state_dict(), "checkpoints/multitask_model.pt")
        print("✅ 多任务模型已保存到 checkpoints/multitask_model.pt")

# 示例数据（实际使用时替换为真实数据）
if __name__ == "__main__":
    # 示例文本（你需要替换为自己的数据文件）
    lm_data = ["人工智能是计算机科学的一个分支。", "Python是一种解释型语言。"] * 100
    qa_data = ["问：什么是机器学习？答：机器学习是让计算机从数据中学习。"] * 50
    sentiment_data = ["这部电影很棒，我很喜欢。"] * 50
    
    trainer = MultiTaskTrainer()
    trainer.train(lm_data, qa_data, sentiment_data, epochs=3)