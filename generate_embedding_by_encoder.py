import os
import json
import torch 
import tqdm
from sentence_transformers import SentenceTransformer
# pip install sentence-transformers

# model_name = 'bert-base-chinese'
# model_name = 'm3e-large'
# model_name = 'bge-base-zh-v1.5'
# model_name = 'bge-large-zh-v1.5'
for model_name in [
    # "gte-large-zh", # in sentence-transformers
    # "acge-large-zh",  # is mean
    # "gte-base-zh", # in sentence-transformers
    # "tao-8k", # same as stella 
    # "IYun-large-zh", # not sure 
    # "tao", # same as stella 
    # "stella-large-zh-v2", # is mean
    # "stella-large-zh", # is mean
    # "bge-large-zh-v1.5", # in sentence-transformers
    # "piccolo-large-zh", # is mean and maybe in sentence-transformers
    "Dmeta-embedding" # in sentence-transformers
]:
    print(model_name)

    embedding_method = f'embedding_by_{model_name}'

    if '#' not in model_name:
        model = SentenceTransformer(f'/shared_home/XXX/{model_name}').cuda()
    else:
        model = SentenceTransformer(model_name.replace("#", "/")).cuda()
    model.eval()
    print(model)

    for data_name in [
        "case-provision", 
        "policy-company",
        "symptom-drug", 
        "objective-course",
    ]:
        print("\n\n========================================================================")
        print(data_name)

        laws = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws.json", 'r'))
        print(len(laws))
        datas = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas.json", 'r'))
        print(len(datas))
        print(datas[0])

        for law in tqdm.tqdm(laws):
            with torch.no_grad():   
                embedding = model.encode(law['law_content']) # (1024,)
            embedding = embedding.reshape(1, embedding.shape[0])
            # print("embedding.shape: ", embedding.shape)
            law_embedding = embedding.tolist()
            # print(law_embedding)
            # print()
            law['embedding'] = law_embedding
        json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)

        print('\n\nlaw embdding done')

        for data in tqdm.tqdm(datas):
            with torch.no_grad():   
                embedding = model.encode(data["context"]) # (1024,)
            embedding = embedding.reshape(1, embedding.shape[0])
            # print("embedding.shape: ", embedding.shape)
            data_embedding = embedding.tolist()
            # print(data_embedding)
            # print()
            data["embedding"] = data_embedding
        json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
        
print('done')