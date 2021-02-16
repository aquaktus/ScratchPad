from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from tqdm import tqdm 

class Environment_Gold_Dataset(Dataset):
    def __init__(self, env, num_samples, **kwargs):
        
        self.env = env
        self.num_samples = num_samples
        
        self.samples = []
        for i in range(self.num_samples):
            gold_sample = env.get_gold()
            input_ids = torch.tensor(gold_sample['gold_state'], dtype=torch.long)
            grad_mask = torch.tensor(gold_sample['trainable_tokens_mask'])
            
            target_policies = torch.zeros((len(input_ids), self.env.action_space.n))
            target_policies[torch.arange(len(input_ids)), input_ids] = 1
            
            new_sample = {
                "input_ids":input_ids[:-1],
                "grad_mask":grad_mask[1:],
                "target_policies":target_policies[1:]
            }
            
            self.samples.append(new_sample)
            env.reset()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def to_dataloader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False):
        dataloader = DataLoader(self, batch_size=batch_size, num_workers=num_workers,\
                           drop_last=False, collate_fn = self.collate, shuffle=shuffle, pin_memory=pin_memory)
        dataloader.__code__ = 0
        return dataloader
        
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        PAD_id = self.env.tokenizer.get_vocab()['[PAD]']
        collated_samples = {}
        
        collated_samples["input_ids"] = torch.nn.utils.rnn.pad_sequence([s['input_ids'] for s in input_samples], padding_value=PAD_id, batch_first=True)
        
        max_seq_len = collated_samples["input_ids"].shape[1]
        action_size = self.env.action_space.n
        collated_samples["target_policies"] = torch.stack([torch.cat([s['target_policies'], torch.zeros(max_seq_len-s['target_policies'].shape[0], action_size)]) for s in input_samples])
        
        
        attention_mask = (collated_samples["input_ids"] != PAD_id)
        
        collated_samples["grad_mask"] = torch.stack([torch.cat([s['grad_mask'], torch.full((max_seq_len-s['grad_mask'].shape[0],), False, dtype=torch.bool)]) for s in input_samples])
        
        collated_samples["grad_mask"] = collated_samples["grad_mask"] & attention_mask
        
        return collated_samples