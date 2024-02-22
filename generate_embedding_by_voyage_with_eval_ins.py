import os
import tqdm
import json
import time
import requests
import voyageai

vo = voyageai.Client(api_key="pa-XXX")

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

def use_api(input_text):
    input_text = input_text[:4096]
    
    retry_cnt = 0
    backoff_time = 30
    
    while retry_cnt <= 5:
        try:
            embedding = vo.embed([input_text], model="voyage-lite-02-instruct", input_type="document").embeddings
            assert len(embedding) == 1 and len(embedding[0]) >= 512, f"embedding length error: {len(embedding)}, this is not a real embedding"
            # print(embedding)    
            # input()
            return embedding
        
        except Exception as e:
            print(f"OpenAIError: {e}.")
            retry_cnt += 1
            print(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            print("retry_cnt:", retry_cnt)

embedding_method = f'embedding_by_voyage-lite-02-instruct_with_eval_ins'

save_i = 0
for data_name in [
    # "case-provision_en", 
    "policy-company_en",
    "symptom-drug_en", 
    "objective-course_en",
]:
    print("\n\n========================================================================")
    print(data_name)

    laws = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws.json", 'r'))
    print(len(laws))
    print(laws[0].keys())
    datas = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas.json", 'r'))
    print(len(datas))
    print(datas[0].keys())
    print(datas[0]['context'])

    for law in tqdm.tqdm(laws, desc="law embedding"):
        law["embedding"] = use_api(law['law_content'] + data_ins_map0[data_name]['law_ins'])

        save_i += 1
        if save_i % 100 == 0:
            print("save law embedding", save_i)
            try:
                json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)
    json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
    
    print('\n\nlaw embdding done')

    for data in tqdm.tqdm(datas, desc="data embedding"):
        data["embedding"] = use_api(data['context'] + data_ins_map0[data_name]['data_ins'])
        
        save_i += 1
        if save_i % 100 == 0:
            print("save data embedding", save_i)
            try:
                json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)
    json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
        
print('done')