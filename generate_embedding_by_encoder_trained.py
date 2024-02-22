import os
import json
import torch 
import tqdm
from utils.SentenceTransformer import SentenceTransformer
# pip install sentence-transformers

data_ins_map0 = {
    "case-provision": {
        "data_ins": "\n以上文字是法律领域中一个案件的描述，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的法律条文，这个表示是：", 
        "law_ins": "\n以上文字是法律领域中的一个法条，请基于自身知识，将其压缩为一个表示，这个表示可以用来与相关的法律案件做匹配，这个表示是：",  
    }, 
    "policy-company": {
        "data_ins": "\n以上文字是金融领域中的一个国家政策，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的公司简介，这个表示是：", 
        "law_ins": "\n以上文字是金融领域中一个上市公司的简介，请基于自身知识，将其压缩为一个表示，这个表示可以与相关金融政策做匹配，这个表示是：",  
    },
    "from_dialogmed_new": {
        "data_ins": "\n以上文字是医疗领域中一个患者的自述症状，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关药物说明，这个表示是：", 
        "law_ins": "\n以上文字是医疗领域中一个药物的说明，请基于自身知识，将其压缩为一个表示，这个表示可以用来与患者的症状做匹配，这个表示是：",  
    },
    "objective-course": {
        "data_ins": "\n以上文字是教育领域中一个培养目标，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关课程简介，这个表示是：", 
        "law_ins": "\n以上文字是教育领域中一个课程的简介，请基于自身知识，将其压缩为一个表示，这个表示可以用来与学生培养目标做匹配，这个表示是：",  
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
    
    "gte-large-zh_manual_1K_no_ins_tuning", # in sentence-transformers
    # "acge-large-zh_manual_1K_no_ins_tuning",  # is mean
    # "gte-base-zh_manual_1K_no_ins_tuning", # in sentence-transformers
    # "tao-8k_manual_1K_no_ins_tuning", # same as stella
    # "IYun-large-zh_manual_1K_no_ins_tuning", # not sure
    # "tao_manual_1K_no_ins_tuning", # same as stella
    "stella-large-zh-v2_manual_1K_no_ins_tuning", # is mean
    # "stella-large-zh_manual_1K_no_ins_tuning", # is mean
    "bge-large-zh-v1.5_manual_1K_no_ins_tuning", # in sentence-transformers
    # "piccolo-large-zh_manual_1K_no_ins_tuning"
]:
    print(model_name)

    embedding_method = f'embedding_by_{model_name}'

    model = SentenceTransformer(f'/shared_home/XXX/{model_name}').cuda()
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
                if "no_ins_tuning" in embedding_method:
                    embedding = model.encode(law['law_content'])
                elif "1K_ins_tuning" in embedding_method:
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
                if "no_ins_tuning" in embedding_method:
                    embedding = model.encode(data["context"])
                elif "1K_ins_tuning" in embedding_method:
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