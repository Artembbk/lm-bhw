import torch
import torch.nn as nn
import wandb

class Trainer():
    def __init__(self, model, optimizer, scheduler, train_dataloader, val_dataloader, total_steps, validate_every, save_checkpoint_every):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_steps = total_steps
        self.validate_every = validate_every
        self.save_checkpoint_every = save_checkpoint_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    def train_step(self, inputs, lengths):
        inputs = inputs.to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.model(inputs[:, :-1], lengths)
        
        loss = self.criterion(logits.view(-1, logits.shape[-1]), inputs[:, 1:].view(-1,))
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for data in self.val_dataloader:
                inputs, lengths = data
                inputs = inputs.to(self.device)
                
                logits = self.model(inputs[:, :-1], lengths)
                loss = self.criterion(logits.view(-1, logits.shape[-1]), inputs[:, 1:].view(-1,))
                
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        self.model.train()
        return total_loss / total_samples
    
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
            
            loss = self.train_step(inputs, lengths)
            wandb.log({"Training Loss": loss})
            
            if step % self.validate_every == 0:
                val_loss = self.validate()
                wandb.log({"Validation Loss": val_loss})
            
            if step % self.save_checkpoint_every == 0:
                torch.save(self.model.state_dict(), f"checkpoint_{step}.pt")
            
            self.scheduler.step()
                
            step += 1

        wandb.finish() 
