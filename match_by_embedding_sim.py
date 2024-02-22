import json
import torch
import os
import tqdm
os.system('clear')


method_names = [
    # "embedding_by_text-embedding-ada-002",
    # "embedding_by_text-embedding-ada-002_with_eval_ins", 
    "embedding_by_text-embedding-3-large",
    "embedding_by_text-embedding-3-large_with_eval_ins",
    
    # "embedding_by_e5-mistral-7b-instruct",
    
    # "embedding_by_baichuan",
    # "embedding_by_baichuan_with_eval_ins",
    
    
    # "embedding_by_gte-large-zh",
    # "embedding_by_acge-large-zh",
    # "embedding_by_gte-base-zh",
    # "embedding_by_tao-8k",
    # "embedding_by_IYun-large-zh",
    # "embedding_by_tao",
    # "embedding_by_stella-large-zh-v2",
    # "embedding_by_stella-large-zh",
    # "embedding_by_bge-large-zh-v1.5",
    # "embedding_by_piccolo-large-zh",
    # "embedding_by_Dmeta-embedding"
    
    # "embedding_by_acge-large-zh_manual_1K_ins_tuning",
    # "embedding_by_gte-base-zh_manual_1K_ins_tuning",
    # "embedding_by_tao-8k_manual_1K_ins_tuning",
    # "embedding_by_IYun-large-zh_manual_1K_ins_tuning",
    # "embedding_by_tao_manual_1K_ins_tuning",
    # "embedding_by_stella-large-zh_manual_1K_ins_tuning",
    
    # "embedding_by_gte-large-zh_manual_1K_ins_tuning",
    # "embedding_by_bge-large-zh-v1.5_manual_1K_ins_tuning",
    # "embedding_by_stella-large-zh-v2_manual_1K_ins_tuning",
    
    # "embedding_by_acge-large-zh_manual_1K_no_ins_tuning",
    # "embedding_by_gte-base-zh_manual_1K_no_ins_tuning",
    # "embedding_by_tao-8k_manual_1K_no_ins_tuning",
    # "embedding_by_IYun-large-zh_manual_1K_no_ins_tuning",
    # "embedding_by_tao_manual_1K_no_ins_tuning",
    # "embedding_by_stella-large-zh_manual_1K_no_ins_tuning",

    # "embedding_by_gte-large-zh_manual_1K_no_ins_tuning",
    # "embedding_by_bge-large-zh-v1.5_manual_1K_no_ins_tuning",
    # "embedding_by_stella-large-zh-v2_manual_1K_no_ins_tuning",

    
    ############ lora train ############    
    
    "embedding_by_Baichuan2-7B-Chat_checkpoint-none",
    
    ## train manual_ins_1K 2 epochc16 
    "embedding_by_Baichuan2-7B-Chat_manual_1K_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final",
    
    "embedding_by_Baichuan2-7B-Chat_manual_1K_ins_0.0_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final", 
    
    "embedding_by_Baichuan2-7B-Chat_manual_1K_no_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final", 
    "embedding_by_Baichuan2-7B-Chat_manual_1K_single_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final",  
    
]


for method_name in method_names:
    
    if "_LAWS_AND_DATAS_" in method_name:
        laws_method_name = method_name.split("_LAWS_AND_DATAS_")[0]
        datas_method_name = method_name.split("_LAWS_AND_DATAS_")[1]
    else:
        laws_method_name = method_name
        datas_method_name = method_name

    for data_name in [
        "policy-company",
        "case-provision", 
        "symptom-drug",
        "objective-course",
    ]:
        print('\n=================================')
        print(laws_method_name)
        print(datas_method_name)
        print(data_name, method_name)

        laws = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{laws_method_name}.json", 'r'))
        print("len(laws):", len(laws))
        print(laws[0].keys())
        datas = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{datas_method_name}.json", 'r'))
        print("len(datas):", len(datas))
        print(datas[0].keys())
        # print(datas[0]['context'])

        results = []
        # note that if more than one embedding for each sentence,  this law['embedding'][0] should be changed
        laws_embedding = torch.tensor([law['embedding'][0] for law in laws]) # law['embedding'] is (1, 768), final is (num_laws, 768)
        print("laws_embedding.shape: ", laws_embedding.shape)
        for data in tqdm.tqdm(datas):
            data_embedding = torch.tensor(data['embedding']) # (1, 768)
            matching = []
            sims = torch.cosine_similarity(data_embedding, laws_embedding, dim=1) # shape is (num_laws)
            # print(sims)
            # print(sims.shape)
            matching = list(zip([law['id'] for law in laws], sims.tolist()))
            # print(matching)
            # print(len(matching))
            matching.sort(key=lambda x: x[1], reverse=True)
            results.append({
                "id": data['id'],
                "context": data['context'], 
                "extended_context": data['extended_context'] if 'extended_context' in data.keys() else None,
                "ann": data['ann'], 
                "pre": matching
            })

        json.dump(results, open(f"/ceph_home/XXX/llm_matching/results/{data_name}/results_by_{method_name}.json", 'w'), ensure_ascii=False, indent=4)
        
print("\n\nall done")