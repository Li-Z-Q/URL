import os
os.system('clear')
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
    # "bge-large-en-v1.5"
    # "ember-v1", 
    "sf_model_e5", 
    # "gte-large", 
    # "stella-base-en-v2", 
    # "e5-large-v2",     
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
        "policy-company_en",
        "case-provision_en", 
        "symptom-drug_en", 
        "objective-course_en"
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