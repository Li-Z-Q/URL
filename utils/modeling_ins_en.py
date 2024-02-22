import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from utils.modeling_llama import LlamaForCausalLM
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = LlamaForCausalLM

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                #  sentence_pooling_method: str = 'last', # use the last token's hidden state as the sentence embedding
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 loss_rate: float = 0,
                 gist_token_ids: Tensor = None,
                 prompt_ids: Tensor = None,
                 use_data_ins: bool = True, # should keep True, because will contral ins by dataset
                 ):
        super().__init__()
        logger.info("Loading model %s", model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.loss_rate = loss_rate
        self.prompt_ids = prompt_ids
        if self.prompt_ids is not None:
            self.prompt_ids = self.prompt_ids.to(self.model.device)
        self.gist_token_ids = gist_token_ids
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.use_data_ins = use_data_ins
        
        self.normlized = normlized
        # self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    # def sentence_embedding(self, hidden_state, mask):
    #     if self.sentence_pooling_method == 'mean':
    #         s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
    #         d = mask.sum(axis=1, keepdim=True).float()
    #         return s / d
    #     elif self.sentence_pooling_method == 'cls':
    #         return hidden_state[:, 0]

    def encode(self, input_ids, attention_mask):
        
        p_reps = None
        if self.gist_token_ids == None:
            psg_out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
            p_reps = psg_out.hidden_states[-1][:, -1, :] # num * 768
        else:
            gist_token_ids = torch.cat([self.gist_token_ids for _ in range(input_ids.shape[0])], dim=0).to(self.model.device)
            gist_mask = torch.ones(gist_token_ids.shape).to(self.model.device)
            total_input_ids = torch.cat((input_ids, gist_token_ids), dim=1)
            total_attention_mask = torch.cat((attention_mask, gist_mask), dim=1)
            psg_out = self.model(input_ids=total_input_ids, attention_mask=total_attention_mask, return_dict=True, output_hidden_states=True)
            p_reps = torch.mean(psg_out.hidden_states[-1][:, input_ids.shape[1]:, :], dim=1) # num * 768
        
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)        
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, 
                query: Dict[str, Tensor] = None, query_ins: Dict[str, Tensor] = None, prompt_for_query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None, corpu_ins: Dict[str, Tensor] = None, prompt_for_corpu: Dict[str, Tensor] = None,
                teacher_score: Tensor = None, loss_type: str = None):
        
        assert self.training, "Model needs to be set to training mode if use forward"
        
        if self.use_data_ins == False:
            query_ins = None
            corpu_ins = None
        
        if loss_type == "debug":
            total_input_ids = torch.cat((query.input_ids, query_ins.input_ids), dim=1)
            total_attention_mask = torch.cat((query.attention_mask, query_ins.attention_mask), dim=1)
            labels = total_input_ids.clone()
            labels[total_attention_mask == 0] = -100
            loss = self.model(
                input_ids=total_input_ids,
                attention_mask=total_attention_mask,
                labels=labels
            ).loss
            
        elif loss_type == 'gist':
            assert passage.input_ids.shape[0] % query.input_ids.shape[0] == 0
            group_size = passage.input_ids.shape[0] // query.input_ids.shape[0] # the num of both of pos and neg for a query
            pos_passage_ids = passage.input_ids[[i*group_size for i in range(query.input_ids.shape[0])], :] 
            pos_passage_mask = passage.attention_mask[[i*group_size for i in range(query.input_ids.shape[0])], :]
            pos_passage_ids = pos_passage_ids[:, -pos_passage_mask.sum(axis=1).max():] # neg is too long cause pos also pad, here cut
            pos_passage_mask = pos_passage_mask[:, -pos_passage_mask.sum(axis=1).max():]
            assert pos_passage_ids.shape == pos_passage_mask.shape
            assert pos_passage_ids.shape[0] == query.input_ids.shape[0] # one pos for each query
            
            # assert (self.prompt_ids == None or prompt_for_corpu == None) # if both not none, use from bash, not use from dataset
            assert (self.prompt_ids != None or prompt_for_corpu != None)
            
            if self.prompt_ids == None:
                prompt_ids = prompt_for_corpu.input_ids
                prompt_mask = prompt_for_corpu.attention_mask
            else:
                prompt_ids = torch.cat([self.prompt_ids for _ in range(query.input_ids.shape[0])], dim=0).to(self.model.device)
                prompt_mask = torch.ones(prompt_ids.shape).to(self.model.device)
            
            if query_ins != None:
                query_with_ins_ids = torch.cat((query.input_ids, query_ins.input_ids), dim=1)
                query_with_ins_mask = torch.cat((query.attention_mask, query_ins.attention_mask), dim=1)
            else:
                query_with_ins_ids = query.input_ids
                query_with_ins_mask = query.attention_mask
            
            if self.gist_token_ids == None:
                total_input_ids = torch.cat((query_with_ins_ids, prompt_ids, pos_passage_ids), dim=1)
                total_attention_mask = torch.cat((query_with_ins_mask, prompt_mask, pos_passage_mask), dim=1)
                gist_position = [
                    query_with_ins_ids.shape[1] - 1, 
                    query_with_ins_ids.shape[1] - 1, 
                ]
            else:
                gist_token_ids =torch.cat([self.gist_token_ids for _ in range(query.input_ids.shape[0])], dim=0).to(self.model.device)
                gist_mask = torch.ones(gist_token_ids.shape).to(self.model.device)
                total_input_ids = torch.cat((query_with_ins_ids, gist_token_ids, prompt_ids, pos_passage_ids), dim=1)
                total_attention_mask = torch.cat((query_with_ins_mask, gist_mask, prompt_mask, pos_passage_mask), dim=1)            
                gist_position = [
                    query_with_ins_ids.shape[1], 
                    query_with_ins_ids.shape[1] + gist_token_ids.shape[1] - 1
                ]
            
            labels = total_input_ids.clone()
            output_start = total_input_ids.shape[1] - pos_passage_ids.shape[1]
            labels[:, :output_start] = -100
            labels[total_attention_mask == 0] = -100
                    
            loss = self.model(
                input_ids=total_input_ids, 
                attention_mask=total_attention_mask, 
                labels=labels, 
                gist_position=gist_position
            ).loss
        
        elif loss_type == 'gist-reverse':
            assert passage.input_ids.shape[0] % query.input_ids.shape[0] == 0
            group_size = passage.input_ids.shape[0] // query.input_ids.shape[0] # the num of both of pos and neg for a query
            pos_passage_ids = passage.input_ids[[i*group_size for i in range(query.input_ids.shape[0])], :] 
            pos_passage_mask = passage.attention_mask[[i*group_size for i in range(query.input_ids.shape[0])], :]
            pos_passage_ids = pos_passage_ids[:, -pos_passage_mask.sum(axis=1).max():] # neg is too long cause pos also pad, here cut
            pos_passage_mask = pos_passage_mask[:, -pos_passage_mask.sum(axis=1).max():]
            assert pos_passage_ids.shape == pos_passage_mask.shape
            assert pos_passage_ids.shape[0] == query.input_ids.shape[0] # one pos for each query
            
            # assert (self.prompt_ids == None or prompt_for_query == None) # if both not none, use from bash, not use from dataset
            assert (self.prompt_ids != None or prompt_for_query != None)
            
            if self.prompt_ids == None:
                prompt_ids = prompt_for_query.input_ids
                prompt_mask = prompt_for_query.attention_mask
            else:
                prompt_ids = torch.cat([self.prompt_ids for _ in range(query.input_ids.shape[0])], dim=0).to(self.model.device)
                prompt_mask = torch.ones(prompt_ids.shape).to(self.model.device)
            
            if corpu_ins != None:
                pos_passage_with_ins_ids = torch.cat((pos_passage_ids, corpu_ins.input_ids), dim=1)
                pos_passage_with_ins_mask = torch.cat((pos_passage_mask, corpu_ins.attention_mask), dim=1)
            else:
                pos_passage_with_ins_ids = pos_passage_ids
                pos_passage_with_ins_mask = pos_passage_mask
            
            if self.gist_token_ids == None:
                total_input_ids = torch.cat((pos_passage_with_ins_ids, prompt_ids, query.input_ids), dim=1)
                total_attention_mask = torch.cat((pos_passage_with_ins_mask, prompt_mask, query.attention_mask), dim=1)
                gist_position = [
                    pos_passage_with_ins_ids.shape[1] - 1, 
                    pos_passage_with_ins_ids.shape[1] - 1, 
                ]
            else:
                gist_token_ids =torch.cat([self.gist_token_ids for _ in range(query.input_ids.shape[0])], dim=0).to(self.model.device)
                gist_mask = torch.ones(gist_token_ids.shape).to(self.model.device)
                total_input_ids = torch.cat((pos_passage_with_ins_ids, gist_token_ids, prompt_ids, query.input_ids), dim=1)
                total_attention_mask = torch.cat((pos_passage_with_ins_mask, gist_mask, prompt_mask, query.attention_mask), dim=1)            
                gist_position = [
                    pos_passage_with_ins_ids.shape[1], 
                    pos_passage_with_ins_ids.shape[1] + gist_token_ids.shape[1] - 1
                ]
            
            labels = total_input_ids.clone()
            output_start = total_input_ids.shape[1] - query.input_ids.shape[1]
            labels[:, :output_start] = -100
            labels[total_attention_mask == 0] = -100
                    
            loss = self.model(
                input_ids=total_input_ids, 
                attention_mask=total_attention_mask, 
                labels=labels, 
                gist_position=gist_position
            ).loss
        
        elif loss_type == 'contrastive':
            
            assert passage.input_ids.shape[0] % query.input_ids.shape[0] == 0
            
            if query_ins != None:
                total_qury_input_ids = torch.cat((query.input_ids, query_ins.input_ids), dim=1)
                total_qury_attention_mask = torch.cat((query.attention_mask, query_ins.attention_mask), dim=1)
            else:
                total_qury_input_ids = query.input_ids
                total_qury_attention_mask = query.attention_mask
            
            if corpu_ins != None:
                group_size = passage.input_ids.shape[0] // query.input_ids.shape[0] # the num of both of pos and neg for a query
                extend_corpu_ins_input_ids = torch.zeros((passage.input_ids.shape[0], corpu_ins.input_ids.shape[1])).to(passage.input_ids.dtype).to(passage.input_ids.device)
                extend_corpu_ins_attention_mask = torch.zeros(extend_corpu_ins_input_ids.shape).to(passage.input_ids.dtype).to(passage.input_ids.device)
                for i in range(query.input_ids.shape[0]):
                    extend_corpu_ins_input_ids[i*group_size:(i+1)*group_size, :] = corpu_ins.input_ids[i, :]
                    extend_corpu_ins_attention_mask[i*group_size:(i+1)*group_size, :] = corpu_ins.attention_mask[i, :]
                total_corpu_input_ids = torch.cat((passage.input_ids, extend_corpu_ins_input_ids), dim=1)
                total_corpu_attention_mask = torch.cat((passage.attention_mask, extend_corpu_ins_attention_mask), dim=1)
            else:
                total_corpu_input_ids = passage.input_ids
                total_corpu_attention_mask = passage.attention_mask
            
            q_reps = self.encode(total_qury_input_ids, total_qury_attention_mask)
            p_reps = self.encode(total_corpu_input_ids, total_corpu_attention_mask)

            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.temperature
            scores = scores.view(q_reps.size(0), -1) # batch_size(num of query) * batch_size*group_size(num of pos and neg together)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0)) # shape is batch_size, each value is pos position for the query
            loss = self.compute_loss(scores, target)

        elif loss_type == 'contrastive-reverse':
            # if self.negatives_cross_device is False:
            #     raise NotImplementedError("negatives_cross_device must be True for contrastive-reverse loss")
            
            assert passage.input_ids.shape[0] % query.input_ids.shape[0] == 0
            group_size = passage.input_ids.shape[0] // query.input_ids.shape[0] # the num of both of pos and neg for a query
            pos_passage_ids = passage.input_ids[[i*group_size for i in range(query.input_ids.shape[0])], :] 
            pos_passage_mask = passage.attention_mask[[i*group_size for i in range(query.input_ids.shape[0])], :]
            pos_passage_ids = pos_passage_ids[:, -pos_passage_mask.sum(axis=1).max():] # neg is too long cause pos also pad, here cut
            pos_passage_mask = pos_passage_mask[:, -pos_passage_mask.sum(axis=1).max():]
            assert pos_passage_ids.shape == pos_passage_mask.shape
            assert pos_passage_ids.shape[0] == query.input_ids.shape[0] # one pos for each query

            if corpu_ins != None:
                pos_with_ins_ids = torch.cat((pos_passage_ids, corpu_ins.input_ids), dim=1)
                pos_with_ins_mask = torch.cat((pos_passage_mask, corpu_ins.attention_mask), dim=1)
            else:
                pos_with_ins_ids = pos_passage_ids
                pos_with_ins_mask = pos_passage_mask
            
            if query_ins != None:
                query_with_ins_ids = torch.cat((query.input_ids, query_ins.input_ids), dim=1)
                query_with_ins_mask = torch.cat((query.attention_mask, query_ins.attention_mask), dim=1)
            else:
                query_with_ins_ids = query.input_ids
                query_with_ins_mask = query.attention_mask
                
            pos_reps = self.encode(pos_with_ins_ids, pos_with_ins_mask)
            query_reps = self.encode(query_with_ins_ids, query_with_ins_mask)
            
            if self.negatives_cross_device:
                pos_reps = self._dist_gather_tensor(pos_reps)
                query_reps = self._dist_gather_tensor(query_reps)
        
            scores = self.compute_similarity(pos_reps, query_reps)
            scores = scores / self.temperature
            scores = scores.view(pos_reps.size(0), -1) # batch_size(num of query) * batch_size*group_size(num of pos and neg together)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            loss = self.compute_loss(scores, target)
        
        else:
            raise NotImplementedError
        
        return EncoderOutput(
            loss=loss,
            scores=None,
            q_reps=None,
            p_reps=None,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
