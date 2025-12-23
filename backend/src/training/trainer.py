from abc import abstractmethod
import logging
import os
import json
import torch
from typing import Type
from tqdm import tqdm
from src.core.config import TrainingConfig
from torch.utils.data import DataLoader
from src.models.base_model import BasicNeuralClassifier
from src.training.utils.metrics import TrainingHistory, calculate_accuracy, calculate_word_level_accuracy

class Trainer:
    def __init__(self, model, exp_name: str,
                 train_loader: DataLoader, val_loader: DataLoader,
                 config: TrainingConfig, logger: logging.Logger, device: str = 'cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = device
        self.logger = logger
        self.exp_name = exp_name

    @abstractmethod
    def predict(model, dataloader, device):
        model.eval() # type: ignore 
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                mask = model._prepare_mask(input_ids, batch.get('mask')) # type: ignore
    
    def validate(self, test=False):
        self.model.eval() 
        total_loss = 0.0
        total_masked_tokens = 0
        all_preds = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = self.model._prepare_mask(input_ids, batch.get('mask'))

                outputs = self.model(input_ids, mask=mask)
                log_probs = outputs['log_probs']
                flat_log_probs = log_probs.view(-1, log_probs.size(-1))
                flat_labels = labels.view(-1)
                flat_mask = mask.view(-1)

                loss = torch.nn.functional.nll_loss(flat_log_probs, flat_labels, reduction='none')
                masked_loss = (loss * flat_mask).sum()
                total_loss += masked_loss.item()

                preds = torch.argmax(log_probs, dim=-1).cpu().tolist()
                num_masked = flat_mask.sum().item()
                total_loss += (loss * flat_mask).sum().item()
                total_masked_tokens += num_masked
                labels_list = labels.cpu().tolist()
                mask_list = mask.cpu().tolist()

                for p, l, m in zip(preds, labels_list, mask_list):
                    if not (len(p) == len(l) == len(m)):
                        self.logger.error('Длины предсказаний, лейблов и масок не совпадают: pred={len(p)}, label={len(l)}, mask={len(m)}')
                        raise ValueError(f"Ошибка длин: pred={len(p)}, label={len(l)}, mask={len(m)}")

                all_preds.extend(preds)
                all_labels.extend(labels_list)
                all_masks.extend(mask_list)

        avg_loss = total_loss / total_masked_tokens if total_masked_tokens > 0 else float('inf')

        token_acc = calculate_accuracy(all_labels, all_preds, all_masks)
        word_acc = calculate_word_level_accuracy(all_labels, all_preds)

        return avg_loss, token_acc, word_acc

    def train(self):
        checkpoint_dir = self.config.checkpoint_dir
        optimizer_name = self.config.optimizer
        weight_decay = self.config.weight_decay
        lr = float(self.config.learning_rate)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.logger.error('Оптимайзер отсуствует. Будет использоваться AdamW')
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_accuracy = 0.0
        best_model_path = ""
        early_stopping_counter = 0
        history = TrainingHistory()
        
        num_epochs = self.config.num_epochs
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = self.model._prepare_mask(input_ids, batch.get('mask'))
                outputs = self.model(input_ids, mask=mask)

                loss = self.model.criterion(
                    outputs['log_probs'].view(-1, outputs['log_probs'].size(-1)),
                    labels.view(-1)
                )
                mask = mask.view(-1)
                loss = (loss * mask).sum() / mask.sum()

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
            self.logger.info(f"EPOCH: {epoch+1} Training Loss: {avg_train_loss:.4f}")

            val_loss, val_accuracy, word_level_accuracy = self.validate()
            self.logger.info(f"EPOCH: {epoch+1} Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Word-level: {word_level_accuracy:.4f}")

            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_word_accuracy': word_level_accuracy
            }
            history += epoch_log

            if word_level_accuracy > best_accuracy:
                best_accuracy = word_level_accuracy
                best_model_path = f"{self.exp_name}_best.pt"
                checkpoint_path = os.path.join(checkpoint_dir, best_model_path)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracy': best_accuracy
                }, checkpoint_path)
                self.logger.info(f"Модель обновлена: {checkpoint_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.config.patience:
                    self.logger.info("Ранняя остановка")
                    self.logger.info(f"="*80)
                    self.logger.info(f"Лучший word-level accuracy: {best_accuracy:.4f}")
                    self.logger.info(f"Путь к модели: {os.path.join(checkpoint_dir, best_model_path)}")
                    self.logger.info(f"="*80)
                    break

        log_file = f'{self.config.log_dir}/{self.exp_name}'
        history.save_json(log_file)
        # with open(f'{self.config.log_dir}/{self.exp_name}.json', 'w', encoding='utf-8') as f:
        #     json.dump(history, f, indent=4, ensure_ascii=False)
        self.logger.info(f"История обучения сохранена: '{log_file}.json")