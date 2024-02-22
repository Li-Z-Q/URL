import os
import tqdm
import json
import time
import requests

def use_baichuan(input_text):
    retry_cnt = 0
    backoff_time = 30
    
    while retry_cnt <= 5:
        try:
            url = "http://api.baichuan-ai.com/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-XXX"
            }
            data = {
                "model": "Baichuan-Text-Embedding",
                "input": input_text
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            # print(response)
            embedding = response.json()['data'][0]['embedding']
            assert len(embedding) >= 512, f"embedding length error: {len(embedding)}, this is not a real embedding"
            # print(embedding)    
            # input()
            return [embedding]
        
        except Exception as e:
            print(f"OpenAIError: {e}.")
            retry_cnt += 1
            print(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            print("retry_cnt:", retry_cnt)

embedding_method = f'embedding_by_baichuan'

save_i = 0
for data_name in [
    # "case-provision", 
    # "policy-company",
    "symptom-drug", 
    "objective-course",
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
        law["embedding"] = use_baichuan(law['law_content'])

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
        data["embedding"] = use_baichuan(data['context'])
        
        save_i += 1
        if save_i % 100 == 0:
            print("save data embedding", save_i)
            try:
                json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)
    json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
        
print('done')