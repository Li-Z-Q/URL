from openai import OpenAI
client = OpenAI(api_key="sk-XXX", base_url="XXX")
global api_id
api_id = 0
api_list = [
    "sk-XXX"
]
import os
import tqdm
import json
import time

# api_name = "text-embedding-ada-002"
api_name = "text-embedding-3-large"
print("api_name:", api_name)

def use_openai(input_text):
    
    input_text = input_text[:4096]
    
    retry_cnt = 0
    backoff_time = 30
    
    while retry_cnt <= 5555:
        try:
            # global api_id
            # api_id += 1
            # api_id %= len(api_list)
            # print("api_id:", api_id)
            # client = OpenAI(api_key=api_list[api_id])
            time.sleep(1)
            
            embedding = client.embeddings.create(
                    model=api_name,
                    input=input_text,
                    encoding_format="float"
                ).data[0].embedding # is a list, len is 1024
            
            assert len(embedding) > 100
            # print(len(embedding)) # 3072 for text-embedding-3-large
            # input() 
            
            return [embedding]
        except Exception as e:
            print(f"OpenAIError: {e}.")
            retry_cnt += 1
            print(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            print("retry_cnt:", retry_cnt)

embedding_method = f'embedding_by_{api_name}'
print("embedding_method:", embedding_method)

save_i = 0
for data_name in [
    "symptom-drug_en", 
    "case-provision_en", 
    "policy-company_en", 
    "objective-course_en" 
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

    for law in tqdm.tqdm(laws, desc=f"{data_name} {api_name} law embedding"):
        law["embedding"] = use_openai(law['law_content'])

        save_i += 1
        if save_i % 10 == 0:
            print("save law embedding", save_i)
            try:
                json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)
    json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
    
    print('\n\nlaw embdding done')

    for data in tqdm.tqdm(datas, desc=f"{data_name} {api_name} data embedding"):
        data["embedding"] = use_openai(data['context'])
        
        save_i += 1
        if save_i % 10 == 0:
            print("save data embedding", save_i)
            try:
                json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)
    json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
        
print('done')