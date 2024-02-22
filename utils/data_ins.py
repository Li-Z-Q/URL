import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import DataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        
        if 'query_ins' not in self.dataset[item].keys():
            return query, passages
        else:
            query_ins = self.dataset[item]['query_ins']
            corpu_ins = self.dataset[item]['corpu_ins']
            prompt_for_corpu = self.dataset[item]['prompt_for_corpu']
            prompt_for_query = self.dataset[item]['prompt_for_query']
            
            return query, passages, query_ins, corpu_ins, prompt_for_corpu, prompt_for_query


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        
        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
            add_special_tokens=False, 
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
            add_special_tokens=False, 
        )
        
        if len(features[0]) == 2:
            return {"query": q_collated, "passage": d_collated}
        else:
            query_ins = [f[2] for f in features]
            corpu_ins = [f[3] for f in features]
            prompt_for_corpu = ['<s>' + f[4] for f in features]
            prompt_for_query = ['<s>' + f[5] for f in features]
            
            query_ins_collated = self.tokenizer(
                query_ins,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=False, 
            )
            corpu_ins_collated = self.tokenizer(
                corpu_ins,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=False, 
            )
            prompt_for_corpu_collated = self.tokenizer(
                prompt_for_corpu,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=False, 
            )
            prompt_for_query_collated = self.tokenizer(
                prompt_for_query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=False, 
            )
            
            return {"query": q_collated, "passage": d_collated, "query_ins": query_ins_collated, "corpu_ins": corpu_ins_collated, "prompt_for_corpu": prompt_for_corpu_collated, "prompt_for_query": prompt_for_query_collated}
