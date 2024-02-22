import os
import tqdm
import json
import time
import requests

data_ins_map0 = {
    "case-provision": {
        "data_ins": "\n以上文字是法律领域中一个案件的描述，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的法律条文，这个表示是：", 
        "law_ins": "\n以上文字是法律领域中的一个法条，请基于自身知识，将其压缩为一个表示，这个表示可以用来与相关的法律案件做匹配，这个表示是：",  
    }, 
    "policy-company": {
        "data_ins": "\n以上文字是金融领域中的一个国家政策，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的公司简介，这个表示是：", 
        "law_ins": "\n以上文字是金融领域中一个上市公司的简介，请基于自身知识，将其压缩为一个表示，这个表示可以与相关金融政策做匹配，这个表示是：",  
    },
    "symptom-drug": {
        "data_ins": "\n以上文字是医疗领域中一个患者的自述症状，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关药物说明，这个表示是：", 
        "law_ins": "\n以上文字是医疗领域中一个药物的说明，请基于自身知识，将其压缩为一个表示，这个表示可以用来与患者的症状做匹配，这个表示是：",  
    },
    "objective-course": {
        "data_ins": "\n以上文字是教育领域中一个培养目标，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关课程简介，这个表示是：", 
        "law_ins": "\n以上文字是教育领域中一个课程的简介，请基于自身知识，将其压缩为一个表示，这个表示可以用来与学生培养目标做匹配，这个表示是：",  
    }
}

def use_baichuan(input_text):
    retry_cnt = 0
    backoff_time = 30
    
    while retry_cnt <= 5:
        try:
            url = "http://api.baichuan-ai.com/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-661299b016a7b551ff286602aad96aa0"
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

embedding_method = f'embedding_by_baichuan_with_eval_ins'

save_i = 0
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
    print(laws[0].keys())
    datas = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas.json", 'r'))
    print(len(datas))
    print(datas[0].keys())
    print(datas[0]['context'])

    for law in tqdm.tqdm(laws, desc="law embedding"):
        law["embedding"] = use_baichuan(law['law_content'] + data_ins_map0[data_name]['law_ins'])

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
        data["embedding"] = use_baichuan(data['context'] + data_ins_map0[data_name]['data_ins'])
        
        save_i += 1
        if save_i % 100 == 0:
            print("save data embedding", save_i)
            try:
                json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)
    json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
        
print('done')