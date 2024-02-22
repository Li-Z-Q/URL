import os
import tqdm
import json
import time
import requests
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from angle_emb import AnglE, Prompts

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

angle = AnglE.from_pretrained('/shared_home/XXX/UAE-Large-V1', pooling_strategy='cls').cuda()
angle.set_prompt(prompt=Prompts.C)
# vec = angle.encode({'text': 'hello world'}, to_numpy=True)
# print(vec)
# vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True)
# print(vecs)

embedding_method = f'embedding_by_UAE-Large-V1'

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

        embeddings = angle.encode({'text': law['law_content']}, to_numpy=True)
        # print(embeddings.shape)
        # input()
        law_embedding = embeddings.tolist()
        law["embedding"] = law_embedding
    json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
    
    print('\nlaw embdding done\n')

    s_time = time.time() # calculte the time as second for each 100 data
    for i, data in enumerate(datas):
        if i % 100 == 0:
            print(f"{i}/{len(datas)}, this 100 data cost {time.time() - s_time} seconds")
            s_time = time.time()

        embeddings = angle.encode({'text': data['context']}, to_numpy=True)
        data_embedding = embeddings.tolist()
        data["embedding"] = data_embedding
    json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
    
print(f'{embedding_method} done, cost minutes: {(time.time() - method_start_time) / 60}') # 20 minute for 3 datasets total
        
print("all done")