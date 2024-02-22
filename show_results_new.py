import json
import random
random.seed(42)
import os
import numpy as np
os.system('clear')
import collections
from sklearn.metrics import ndcg_score
import pytrec_eval
import copy

method_names = [
    "bm25", 
    # "bm25_after_tokenize",
    
    "embedding_by_baichuan",
    "embedding_by_baichuan_with_eval_ins",
    "embedding_by_text-embedding-ada-002",
    "embedding_by_text-embedding-ada-002_with_eval_ins", 
    "embedding_by_text-embedding-3-large",
    "embedding_by_text-embedding-3-large_with_eval_ins",
    
    "embedding_by_e5-mistral-7b-instruct",
    
    "embedding_by_Dmeta-embedding",
    "embedding_by_acge-large-zh",
    "embedding_by_gte-large-zh",
    "embedding_by_gte-base-zh",
    "embedding_by_tao-8k",
    "embedding_by_IYun-large-zh",
    "embedding_by_tao",
    "embedding_by_stella-large-zh-v2",
    "embedding_by_stella-large-zh",
    "embedding_by_bge-large-zh-v1.5",
    "embedding_by_piccolo-large-zh",
    
    # "embedding_by_acge-large-zh_manual_1K_ins_tuning",
    # "embedding_by_gte-base-zh_manual_1K_ins_tuning",
    # "embedding_by_tao-8k_manual_1K_ins_tuning",
    # "embedding_by_IYun-large-zh_manual_1K_ins_tuning",
    # "embedding_by_tao_manual_1K_ins_tuning",
    # "embedding_by_stella-large-zh_manual_1K_ins_tuning",
    
    # "embedding_by_gte-large-zh_manual_1K_ins_tuning",
    # "embedding_by_bge-large-zh-v1.5_manual_1K_ins_tuning",
    # "embedding_by_stella-large-zh-v2_manual_1K_ins_tuning",
    
    "embedding_by_acge-large-zh_manual_1K_no_ins_tuning",
    "embedding_by_gte-base-zh_manual_1K_no_ins_tuning",
    "embedding_by_tao-8k_manual_1K_no_ins_tuning",
    "embedding_by_IYun-large-zh_manual_1K_no_ins_tuning",
    "embedding_by_tao_manual_1K_no_ins_tuning",
    "embedding_by_stella-large-zh_manual_1K_no_ins_tuning",

    "embedding_by_gte-large-zh_manual_1K_no_ins_tuning",
    "embedding_by_bge-large-zh-v1.5_manual_1K_no_ins_tuning",
    "embedding_by_stella-large-zh-v2_manual_1K_no_ins_tuning",

    
    "embedding_by_Baichuan2-7B-Chat_checkpoint-none",
    
    "embedding_by_Baichuan2-7B-Chat_manual_1K_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final",
    
    "embedding_by_Baichuan2-7B-Chat_manual_1K_ins_0.0_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final", 
    "embedding_by_Baichuan2-7B-Chat_manual_1K_no_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final", 
    "embedding_by_Baichuan2-7B-Chat_manual_1K_single_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#_checkpoint-final",  
    
    

]




metrics = [
    'ndcg_cut_5', 'ndcg_cut_10', "ndcg_cut_20", "ndcg_cut_50", "ndcg_cut_100",
    'map_cut_5', 'map_cut_10', "map_cut_20", "map_cut_50", "map_cut_100",
    "recall_5", "recall_10", "recall_20", "recall_50", "recall_100",
    "P_5", "P_10", "P_20", "P_50", "P_100",
]

# metrics = [
#     "ndcg_cut_20", 
#     "map_cut_20", 
# ]

file_names = {
    "policy-company": [], # 709
    "case-provision": [], # 3627 laws
    "symptom-drug": [], # 15830
    "objective-course": []
}

for data_name in list(file_names.keys()):
    
    laws = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws.json", 'r'))
    # print("len(laws): ", len(laws))
    # print(laws[0].keys())
    datas = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas.json", 'r'))
    # print("len(datas): ", len(datas))
    # print(datas[0].keys())
    
    file_names[data_name] = np.zeros((len(method_names), len(metrics)))
    
    for i, method_name in enumerate(method_names):
            
        results = json.load(open(f"/ceph_home/XXX/llm_matching/results/{data_name}/results_by_{method_name}.json", 'r'))
        # print("len(results): ", len(results))
        # print(results[0].keys())
        
        print('\n\n=================================')
        print(method_name, data_name)

        qrel = {}
        for data in datas: # it is easy to contral trick because use ann from original datas rather than results files
            qrel[str(data['id'])] = {}
            for law_id in data['ann']:
                qrel[str(data['id'])][str(law_id)] = 1
        
        run = {}
        for data in results:
            run[str(data['id'])] = {}
            for law_id, score in data['pre']:
                run[str(data['id'])][str(law_id)] = score

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        scores = evaluator.evaluate(run)

        
        for query_id in scores.keys():
            for m, metric in enumerate(metrics):
                file_names[data_name][i][m] += scores[query_id][metric]

    file_names[data_name] = 100 * file_names[data_name] / len(scores)        
         
print("all done")

print(method_names)
print(metrics)
# numpy to csv and save
for data_name in list(file_names.keys()):
    print(data_name)
    print(file_names[data_name])
    # np.savetxt(f"{data_name}.csv", file_names[data_name], delimiter=",", fmt='%1.2f', header=",".join(metrics), comments='')
    # np not work, i want to save to csv contrain method name and file_names[data_name]
    # so i use pandas
    import pandas as pd
    df = pd.DataFrame(file_names[data_name], columns=metrics)
    df.index = method_names
    df.to_csv(f"{data_name}.csv")
    
print("all done")