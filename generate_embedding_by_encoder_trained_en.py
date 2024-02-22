import os
import json
import torch 
import tqdm
from utils.SentenceTransformer import SentenceTransformer
# pip install sentence-transformers

data_ins_map0 = {
    "case-provision_en": {
        "data_ins": "\nAbove is a legal case description in law domain, compress it into an embedding representation based on your knowledge, this embedding can be used to link relevant legal provisions, the embedding is:",
        "law_ins": "\nAbove is a legal provision in law domain, compress it into an embedding representation based on your knowledge, this embedding can be used to match relevant legal cases, the embedding is:",
    }, 
    "policy-company_en": {
        "data_ins": "\nAbove is a government policy in finance domain, compress it into an embedding representation based on your knowledge, this embedding can be used to link relevant company profiles, the embedding is:",
        "law_ins": "\nAbove is a company profile in finance domain, compress it into an embedding representation based on your knowledge, this embedding can be used to match relevant government policies, the embedding is:",
    },
    "symptom-drug_en": {
        "data_ins": "\nAbove is a patient symptom in medical domain, compress it into an embedding representation based on your knowledge, this embedding can be used to link relevant drug descriptions, the embedding is:",
        "law_ins": "\nAbove is a drug description in medical domain, compress it into an embedding representation based on your knowledge, this embedding can be used to match relevant patient symptoms, the embedding is:",
    },
    "objective-course_en": {
        "data_ins": "\nAbove is a training objective in education domain, compress it into an embedding representation based on your knowledge, this embedding can be used to link relevant course introductions, the embedding is:",
        "law_ins": "\nAbove is a course introduction in education domain, compress it into an embedding representation based on your knowledge, this embedding can be used to match relevant training objectives, the embedding is:",
    }
}

for model_name in [
    # "gte-large-zh_manual_1K_ins_tuning", # in sentence-transformers
    # "acge-large-zh_manual_1K_ins_tuning",  # is mean
    # "gte-base-zh_manual_1K_ins_tuning", # in sentence-transformers
    # "tao-8k_manual_1K_ins_tuning", # same as stella
    # "IYun-large-zh_manual_1K_ins_tuning", # not sure
    # "tao_manual_1K_ins_tuning", # same as stella
    # "stella-large-zh-v2_manual_1K_ins_tuning", # is mean
    # "stella-large-zh_manual_1K_ins_tuning", # is mean
    # "bge-large-zh-v1.5_manual_1K_ins_tuning", # in sentence-transformers
    
    "bge-large-en-v1.5_manual_1K_no_ins_en_tuning", # in sentence-transformers
    "gte-large_manual_1K_no_ins_en_tuning",  # is mean
    "stella-base-en-v2_manual_1K_no_ins_en_tuning", # in sentence-transformers
]:
    print(model_name)

    embedding_method = f'embedding_by_{model_name}'

    model = SentenceTransformer(f'/shared_home/XXX/{model_name}').cuda()
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
                if "no_ins_en_tuning" in embedding_method:
                    embedding = model.encode(law['law_content'])
                elif "1K_ins_en_tuning" in embedding_method:
                    embedding = model.encode(law['law_content'] + data_ins_map0[data_name]['law_ins']) # (1024,)
                else:
                    raise ValueError
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
                if "no_ins_en_tuning" in embedding_method:
                    embedding = model.encode(data["context"])
                elif "1K_ins_en_tuning" in embedding_method:
                    embedding = model.encode(data["context"] + data_ins_map0[data_name]['data_ins']) # (1024,)
                else:
                    raise ValueError
            embedding = embedding.reshape(1, embedding.shape[0])
            # print("embedding.shape: ", embedding.shape)
            data_embedding = embedding.tolist()
            # print(data_embedding)
            # print()
            data["embedding"] = data_embedding
        json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
        
print('done')