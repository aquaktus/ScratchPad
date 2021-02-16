import collections
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning.core.lightning import LightningModule

from transformers import LongformerForSequenceClassification, LongformerModel, BertModel, BertTokenizer, BertForSequenceClassification, \
                         get_linear_schedule_with_warmup, BertConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation_utils import GenerationMixin
import numpy as np
from transformers.optimization import AdamW

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, make_proba_distribution
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from stable_baselines3.common.utils import get_device, is_vectorized_observation
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

base_config = BertConfig()
base_config.num_attention_heads = 8
base_config.hidden_size = 256
base_config.num_hidden_layers = 12
base_config.intermediate_size = 1024
base_config.n_ctx = 512
base_config.n_positions = 512
base_config.is_decoder = True
base_config.temp = 1
base_config.position_embedding_type = 'relative_key_query'


class CausalBERTFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.MultiDiscrete, action_space: gym.spaces.Discrete, config=None, pad_id=0):
        super(CausalBERTFeatureExtractor, self).__init__(observation_space, config.hidden_size)
        self.config = config
        self.config.vocab_size = action_space.n
        self.transformer = BertModel(config)
        self.pad_id = pad_id

    def forward(self, observations: torch.Tensor):
        batch_size = observations.shape[0]
        observations = observations.reshape(batch_size, -1, self.config.vocab_size)
        input_ids = observations.argmax(dim=-1)
        
        attention_mask = (input_ids != self.pad_id)
        sequence_lengths = torch.sum(attention_mask, dim=1)
        last_token_positions = sequence_lengths-1
        
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        
        last_token_hidden_states = hidden_states[torch.arange(batch_size),last_token_positions,:]
                
        return last_token_hidden_states
    
        
class CausalBertLMPolicyWrapper(LightningModule, GenerationMixin):
    def __init__(self, policy, pad_id=0, **kwargs):
        super().__init__(**kwargs)
        
        self.policy = policy
        self.pad_id = pad_id
        self.action_head = policy.action_net
        self.value_head = policy.value_net
        self.vocab_size = policy.action_space.n
        self.config = policy.features_extractor.config
        self.to(policy.device)
        self.dropout = nn.Dropout(self.policy.features_extractor.config.hidden_dropout_prob)
    
    def forward(self, input_ids, **kwargs):
        "ids: int tensor([batch_size, sequence_length])"
        batch_size, seq_len = input_ids.shape
        
        attention_mask = (input_ids != self.pad_id)
        tarnsformer_outputs = self.policy.features_extractor.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = tarnsformer_outputs[0] # [batch_sz, seq_len, hidden_dim]
        hidden_states = self.dropout(hidden_states)
        lm_logits = self.action_head(hidden_states)
        return CausalLMOutput(logits=lm_logits)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        grad_mask = batch['grad_mask']
        target_policies = batch["target_policies"]
        batch_size = input_ids.shape[0]
        
        outputs = self(input_ids)
        policy_logits = outputs.logits
        
        target_ids = batch['target_policies'].argmax(-1)
        
        loss = nn.CrossEntropyLoss()
#         policy_loss = loss(policy_logits[grad_mask].view(-1, self.vocab_size), target_ids[grad_mask].view(-1))
        
        policy_loss = -torch.dot(target_policies[grad_mask].view(-1), torch.log(policy_logits[grad_mask].softmax(-1)).view(-1))/grad_mask.sum()
        loss = policy_loss
            
        return {"loss":loss, 'log': {'train_loss': loss}}
    
    def configure_optimizers(self):
        self.lr=0.00005
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=10000, epochs=1)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=15000)
        return optimizer#, [scheduler]