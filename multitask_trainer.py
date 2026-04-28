# multitask_trainer.py —— Multi-task Learning Trainer
import torch, random
from config import Config
from model import MiniChat
from lora import apply_lora_to_model, mark_only_lora_as_trainable
import sentencepiece as spm

cfg = Config()
device = cfg.device

sp = spm.SentencePieceProcessor()
sp.load(f"tokenizer/{cfg.tokenizer_prefix}.model")
vocab_size = sp.get_piece_size()

class MultiTaskTrainer:
    def __init__(self):
        self.model = MiniChat(vocab_size).to(device)
        apply_lora_to_model(self.model)
        mark_only_lora_as_trainable(self.model)
        self.task_weights = {"lm": 0.5, "qa": 0.3, "sentiment": 0.2}
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.learning_rate)

    def prepare_sample(self, text, task_type="lm"):
        if task_type == "qa": text = f"Q&A: {text}"
        elif task_type == "sentiment": text = f"Sentiment: {text}"
        ids = sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        if len(ids) > cfg.max_seq_len: ids = ids[:cfg.max_seq_len-1] + [sp.eos_id()]
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        targets = input_ids.clone()
        return input_ids, targets

    def train_step(self, batch_texts, task_type):
        self.model.train()
        total_loss = 0
        for text in batch_texts:
            input_ids, targets = self.prepare_sample(text, task_type)
            _, loss, _, _ = self.model(input_ids, targets=targets)
            weighted_loss = self.task_weights[task_type] * loss
            self.optimizer.zero_grad(); weighted_loss.backward(); self.optimizer.step()
            total_loss += weighted_loss.item()
        return total_loss / len(batch_texts)

    def train(self, lm_data, qa_data, sentiment_data, epochs=5):
        print(f"🚀 Starting multi-task learning...")
        for epoch in range(1, epochs+1):
            all_tasks = []
            all_tasks.extend([("lm", t) for t in lm_data])
            all_tasks.extend([("qa", t) for t in qa_data])
            all_tasks.extend([("sentiment", t) for t in sentiment_data])
            random.shuffle(all_tasks)
            total_loss = 0
            batch_size = 8
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i:i+batch_size]
                batch_texts = [item[1] for item in batch]
                task_type = batch[0][0]
                total_loss += self.train_step(batch_texts, task_type)
            print(f"Epoch {epoch}, Avg Loss: {total_loss/(len(all_tasks)//batch_size):.4f}")
        torch.save(self.model.state_dict(), "checkpoints/multitask_model.pt")
        print("✅ Multi-task model saved.")

if __name__ == "__main__":
    lm_data = ["AI is a branch of computer science.", "Python is an interpreted language."] * 100
    qa_data = ["Q: What is machine learning? A: Machine learning enables computers to learn from data."] * 50
    sentiment_data = ["This movie is great, I love it!"] * 50
    trainer = MultiTaskTrainer()
    trainer.train(lm_data, qa_data, sentiment_data, epochs=3)