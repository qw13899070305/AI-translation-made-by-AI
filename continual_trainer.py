# continual_trainer.py —— Continual Learning Trainer
import torch, random
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
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.learning_rate)
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def encode_text(self, text):
        ids = sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        if len(ids) > cfg.max_seq_len: ids = ids[:cfg.max_seq_len-1] + [sp.eos_id()]
        return torch.tensor(ids, dtype=torch.long)

    def add_to_buffer(self, texts):
        for text in texts:
            if len(self.replay_buffer) < self.replay_buffer.maxlen: self.replay_buffer.append(text)

    def sample_replay(self, batch_size=4):
        if len(self.replay_buffer) < batch_size: return list(self.replay_buffer)
        return random.sample(list(self.replay_buffer), batch_size)

    def learn_task(self, task_name, data_texts, epochs=3, replay_weight=0.3):
        print(f"📚 Learning task: {task_name}, data size: {len(data_texts)}")
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(data_texts)
            for i in range(0, len(data_texts), 8):
                batch_texts = data_texts[i:i+8]
                new_loss = 0
                for text in batch_texts:
                    input_ids = self.encode_text(text).unsqueeze(0).to(device)
                    targets = input_ids.clone()
                    _, loss, _, _ = self.model(input_ids, targets=targets)
                    new_loss += loss
                new_loss /= len(batch_texts)
                replay_texts = self.sample_replay(batch_size=4)
                replay_loss = 0
                if replay_texts:
                    for text in replay_texts:
                        input_ids = self.encode_text(text).unsqueeze(0).to(device)
                        targets = input_ids.clone()
                        _, loss, _, _ = self.model(input_ids, targets=targets)
                        replay_loss += loss
                    replay_loss /= len(replay_texts)
                total = new_loss + replay_weight * replay_loss
                self.optimizer.zero_grad(); total.backward(); self.optimizer.step()
                total_loss += total.item()
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(data_texts):.4f}")
        self.add_to_buffer(data_texts[:500])
        print(f"✅ Task {task_name} completed, buffer size: {len(self.replay_buffer)}")

    def save_model(self, path="checkpoints/continual_model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")

if __name__ == "__main__":
    trainer = ContinualTrainer()
    task1_data = ["Hello, I am an AI assistant.", "The weather is nice today."] * 100
    trainer.learn_task("Basic Conversation", task1_data, epochs=3)
    task2_data = ["Python is an interpreted language.", "Machine learning requires a lot of data."] * 100
    trainer.learn_task("Programming Knowledge", task2_data, epochs=3)
    trainer.save_model()