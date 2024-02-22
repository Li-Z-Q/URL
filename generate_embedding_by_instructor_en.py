import os
import tqdm
import json
import time
import requests
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('hkunlp/instructor-xl')

data_ins_map0 = {
    "case-provision_en": {
        "data_ins": "\nRepresent the legal case:",
        "law_ins": "\nRepresent the legal provision:",
    }, 
    "policy-company_en": {
        "data_ins": "\nRepresent the government policy:",
        "law_ins": "\nRepresent the company profile:",
    },
    "symptom-drug_en": {
        "data_ins": "\nRepresent the patient symptom:",
        "law_ins": "\nRepresent the drug description:",
    },
    "objective-course_en": {
        "data_ins": "\nRepresent the training objective:",
        "law_ins": "\nRepresent the course introduction:",
    }
}

embedding_method = f'embedding_by_instructor-xl'


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

        embeddings = model.encode([[data_ins_map0[data_name]['law_ins'], law['law_content']]])
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
        
        embeddings = model.encode([[data_ins_map0[data_name]['data_ins'], data['context']]])
        # print("embeddings.shape:", embeddings.shape)

        data_embedding = embeddings.tolist()
        data["embedding"] = data_embedding
    json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
    
print(f'{embedding_method} done, cost minutes: {(time.time() - method_start_time) / 60}') # 20 minute for 3 datasets total
        
print("all done")