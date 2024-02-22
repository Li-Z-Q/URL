import os
import tqdm
import json
import time
import requests
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


data_ins_map0 = {
    "case-provision": {
        "data_ins": "这是法律领域中一个案件的描述，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的法律条文", 
    }, 
    "policy-company": {
        "data_ins": "这是金融领域中的一个国家政策，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的公司简介", 
    },
    "symptom-drug": {
        "data_ins": "这是医疗领域中一个患者的自述症状，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关药物说明", 
    },
    "objective-course": {
        "data_ins": "这是教育领域中一个培养目标，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关课程简介", 
    }
}

embedding_method = f'embedding_by_e5-mistral-7b-instruct'

tokenizer = AutoTokenizer.from_pretrained('/shared_home/XXX/e5-mistral-7b-instruct')
model = AutoModel.from_pretrained('/shared_home/XXX/e5-mistral-7b-instruct', torch_dtype=torch.float16).cuda()
model.eval()
max_length = 512
print(model.config)

method_start_time = time.time()
for data_name in [
    "policy-company",
    "case-provision", 
    "symptom-drug", 
    "objective-course"
]: # about 10 minutes for total three datasets
    print("\n\n=========================")
    print(data_name)

    laws = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws.json", 'r'))
    print(len(laws))
    print(laws[0].keys())
    datas = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas.json", 'r'))
    print(len(datas))
    print(datas[0].keys())
    print(datas[0]['context'])

    s_time = time.time() # calculte the time as second for each 100 data
    for i, law in enumerate(laws):
        if i % 100 == 0:
            print(f"{i}/{len(laws)}, this 100 data cost {time.time() - s_time} seconds")
            s_time = time.time()

        input_texts = [law['law_content']]
        
        batch_dict = tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
        batch_dict = batch_dict.to(model.device)
        
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # print("embeddings.shape:", embeddings.shape)

        law_embedding = embeddings.tolist()
        law["embedding"] = law_embedding
    json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
    
    print('\nlaw embdding done\n')

    s_time = time.time() # calculte the time as second for each 100 data
    for i, data in enumerate(datas):
        if i % 100 == 0:
            print(f"{i}/{len(datas)}, this 100 data cost {time.time() - s_time} seconds")
            s_time = time.time()

        input_texts = [get_detailed_instruct(data_ins_map0[data_name]['data_ins'], data['context'])]
        
        batch_dict = tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
        batch_dict = batch_dict.to(model.device)

        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # print("embeddings.shape:", embeddings.shape)

        data_embedding = embeddings.tolist()
        data["embedding"] = data_embedding
    json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
    
print(f'{embedding_method} done, cost minutes: {(time.time() - method_start_time) / 60}') # 20 minute for 3 datasets total
        
print("all done")