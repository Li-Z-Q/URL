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
    "case-provision_en": {
        "data_ins": "Given a legal case, find legal provisions that explain the case", 
    }, 
    "policy-company_en": {
        "data_ins": "Given a government policy, find profile of companies that are affected by the policy",
    },
    "symptom-drug_en": {
        "data_ins": "Given a patient symptom, find description of drugs that resolve the symptom",
    },
    "objective-course_en": {
        "data_ins": "Given a training objective, find introduction of courses that support the description",
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
    "policy-company_en",
    "case-provision_en", 
    "symptom-drug_en", 
    "objective-course_en"
]: 
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