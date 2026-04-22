# continual_trainer.py —— 持续学习训练器（带经验回放）
import torch
import random
from collections import deque
from config import Config
from model import MiniChat
from lora import apply_lora_to_model, mark_only_lora_as_trainable
import sentencepiece as spm

cfg = Config()
device = cfg.device

sp = spm.SentencePieceProcessor()
sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
vocab_size = sp.get_piece_size()

class ContinualTrainer:
    def __init__(self, replay_buffer_size=2000):
        self.model = MiniChat(vocab_size).to(device)
        apply_lora_to_model(self.model)
        mark_only_lora_as_trainable(self.model)
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate
        )
        
        self.replay_buffer = deque(maxlen=replay_buffer_size)
    
    def encode_text(self, text):
        """将文本编码为模型输入"""
        ids = sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        if len(ids) > cfg.max_seq_len:
            ids = ids[:cfg.max_seq_len-1] + [sp.eos_id()]
        return torch.tensor(ids, dtype=torch.long)
    
    def add_to_buffer(self, texts):
        """将文本样本加入回放缓冲区"""
        for text in texts:
            if len(self.replay_buffer) < self.replay_buffer.maxlen:
                self.replay_buffer.append(text)
    
    def sample_replay(self, batch_size=4):
        """从缓冲区随机采样"""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        return random.sample(list(self.replay_buffer), batch_size)
    
    def learn_task(self, task_name, data_texts, epochs=3, replay_weight=0.3):
        """学习一个新任务，同时回放旧知识"""
        print(f"📚 学习任务: {task_name}, 数据量: {len(data_texts)}")
        
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(data_texts)
            
            for i in range(0, len(data_texts), 8):
                batch_texts = data_texts[i:i+8]
                
                # 新任务训练
                new_loss = 0
                for text in batch_texts:
                    input_ids = self.encode_text(text).unsqueeze(0).to(device)
                    targets = input_ids.clone()
                    _, loss, _ = self.model(input_ids, targets=targets)
                    new_loss += loss
                new_loss /= len(batch_texts)
                
                # 经验回放
                replay_texts = self.sample_replay(batch_size=4)
                replay_loss = 0
                if replay_texts:
                    for text in replay_texts:
                        input_ids = self.encode_text(text).unsqueeze(0).to(device)
                        targets = input_ids.clone()
                        _, loss, _ = self.model(input_ids, targets=targets)
                        replay_loss += loss
                    replay_loss /= len(replay_texts)
                
                total = new_loss + replay_weight * replay_loss
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()
                total_loss += total.item()
            
            print(f"  Epoch {epoch+1}, 损失: {total_loss/len(data_texts):.4f}")
        
        # 将新任务数据加入回放缓冲区
        self.add_to_buffer(data_texts[:500])  # 只保留部分节省内存
        print(f"✅ 任务 {task_name} 学习完成，缓冲区大小: {len(self.replay_buffer)}")
    
    def save_model(self, path="checkpoints/continual_model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"💾 模型已保存到 {path}")

# 使用示例
if __name__ == "__main__":
    trainer = ContinualTrainer()
    
    # 任务1：学习基础对话
    task1_data = ["你好，我是AI助手。", "今天天气真好。"] * 100
    trainer.learn_task("基础对话", task1_data, epochs=3)
    
    # 任务2：学习编程知识（此时会回放任务1的数据）
    task2_data = ["Python是一种解释型语言。", "机器学习需要大量数据。"] * 100
    trainer.learn_task("编程知识", task2_data, epochs=3)
    
    trainer.save_model()