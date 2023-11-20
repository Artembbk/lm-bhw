import json
import argparse
from dataset.dataset import TinyStoriesDataset
import importlib
from collate.collate import collate_fn
import torch
from torch.utils.data import DataLoader
from trainer.trainer import Trainer

def initialize_class(class_name, args):
    module_name = f"models.{class_name}"
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**args)
    return instance


def main(config):
    dataset = TinyStoriesDataset(**config['dataset'])
    val_size = config['val_size']
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset,collate_fn=collate_fn, pin_memory=True, **config['dataloaders']["train"])
    val_dataloader = DataLoader(val_dataset,collate_fn=collate_fn, pin_memory=True, **config['dataloaders']["val"])
    
    model_args = config['model']['args']
    model_args['vocab_size'] = config['dataset']['vocab_size']
    model = initialize_class(config['model']['name'], model_args)
    
    optimizer_cls = getattr(torch.optim, config['optimizer']['name'])
    optimizer = optimizer_cls(model.parameters(), **config['optimizer']['args'])

    scheduler_cls = getattr(torch.optim.lr_scheduler, config['scheduler']['name'])
    scheduler = scheduler_cls(optimizer, **config['scheduler']['args'])

    trainer = Trainer(model, optimizer, scheduler, train_dataloader, val_dataloader, **config['training_args'])
    trainer.train()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description='Argument for train')
    
    parser.add_argument('--config', type=str,
                    help='path to config')
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    main(config)

    

    
    
