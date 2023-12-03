import torch
import torch.nn as nn
from torch import Tensor
import wandb
from torch.nn import functional as F
from metrics import perplexity
from utils import create_non_special_mask
from generate import generate_argmax, generate_nucleus
import pandas as pd

class Trainer():
    def __init__(self, model, tokenizer, optimizer, scheduler, train_dataloader, val_dataloader, total_steps, validate_every, save_checkpoint_every):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_steps = total_steps
        self.validate_every = validate_every
        self.save_checkpoint_every = save_checkpoint_every
        self.tokenizer = tokenizer

    def step(self, inputs, lengths, train):
        inputs = inputs.to(self.device)
        
        if train:
            self.optimizer.zero_grad()
        logits = self.model(inputs[:, :-1], lengths)
        perpl = perplexity(inputs, logits, create_non_special_mask(lengths, 256).to(self.device))
        
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), inputs[:, 1:].reshape(-1,))
        if train:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        
        return loss.item(), perpl

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        total_perpl = 0
        
        with torch.no_grad():
            for data in self.val_dataloader:
                inputs, lengths = data
                loss, perpl = self.step(inputs, lengths, False)
                total_perpl += perpl
                total_loss += loss * inputs.size(0)
                total_samples += inputs.size(0)
        
        self.model.train()
        return total_loss / total_samples, total_perpl / len(self.val_dataloader)
    
    def train(self):
        # Настройка WandB
        wandb.init(project='bhw-llm-tiny')
        wandb.watch(self.model)

        # Обучение
        self.model.train()
        step = 0

        for inputs, lengths in self.train_dataloader:
            if step >= self.total_steps:
                break
            
            loss, perpl = self.step(inputs, lengths, True)
            
            if step % self.validate_every == 0:
                val_loss, val_perpl = self.validate()

                wandb.log({"Validation Loss": val_loss})
                wandb.log({"Validation Perplexity": val_perpl})
                self.log_predictions(10)
            
            if step % self.save_checkpoint_every == 0:
                torch.save(self.model.state_dict(), f"checkpoint_{step}.pt")

            if step % 100 == 0:  # Логгирование каждые 100 шагов
                wandb.log({"Training Loss": loss})
                wandb.log({"Training Perplexity": perpl})
                for name, param in self.model.named_parameters():
                    wandb.log({f"Model Parameter {name}": param.clone().cpu().detach().numpy()})

            step += 1

        wandb.finish() 

    def log_predictions(self, num):
        inputs, _ = next(iter(self.val_dataloader))
        inputs = inputs[:num, :2]
        argmax_text = self.tokenizer.decode_batch(generate_argmax(self.model, self.tokenizer, self.device, prefix=inputs, max_len=16).cpu().numpy())
        nucleus_text = self.tokenizer.decode_batch(generate_nucleus(self.model, self.tokenizer, self.device, prefix=inputs, max_len=16).cpu().numpy())
        data = {'Texts': argmax_text}
        df = pd.DataFrame(data)
        wandb.log({"Argamx Texts": wandb.Table(dataframe=df)})
        data = {'Texts': nucleus_text}
        df = pd.DataFrame(data)
        wandb.log({"Nucleus Texts": wandb.Table(dataframe=df)})