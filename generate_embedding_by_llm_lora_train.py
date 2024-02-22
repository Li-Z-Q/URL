import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.system("CUDA_VISIBLE_DEVICES=0")
import json
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.peft_model import PeftModel
# from peft import PeftModel
import time

# base="Base"
# train_data="zh_middle"

# base_model_path = f'/ceph_home/XXX/Baichuan2-7B-{base}'
# tokenizer = AutoTokenizer.from_pretrained(f'{base_model_path}', use_fast=False, trust_remote_code=True)
# print(tokenizer)

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

data_ins_map1 = { # is used in 2.5K data's eval at start
    "case-provision": {
        "data_ins": "\n以上文字是一个案件的描述，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的法律条文，这个表示是：", 
        "law_ins": "\n以上文字是一条法律条文，请基于自身知识，将其压缩为一个表示，这个表示可以用来与相关的法律案件做匹配，这个表示是：",  
    }, 
    "policy-company": {
        "data_ins": "\n以上文字是一个金融政策，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的公司简介，这个表示是：", 
        "law_ins": "\n以上文字是一个上市公司的简介，请基于自身知识，将其压缩为一个表示，这个表示可以与相关的金融政策做匹配，这个表示是：",  
    },
    "from_dialogmed_new": {
        "data_ins": "\n以上文字是一个患者自述的症状，请基于自身知识，将其压缩为一个表示，这个表示可以用来检索相关的药物说明书，这个表示是：", 
        "law_ins": "\n以上文字是一个药物的说明书，请基于自身知识，将其压缩为一个表示，这个表示可以用来与患者的症状做匹配，这个表示是：",  
    },
}

data_ins_map2 = { # 
    "case-provision": {
        "data_ins": "\n以上文字是案件的描述，请基于法律知识，将其压缩为一个表示，用来检索相关的法律条文，这个表示是：", 
        "law_ins": "\n以上文字是法律条文，请基于法律知识，将其压缩为一个表示，用来与相关的法律案件做匹配，这个表示是：",  
    }, 
    "policy-company": {
        "data_ins": "\n以上文字是金融政策，请基于金融知识，将其压缩为一个表示，用来检索相关的公司简介，这个表示是：", 
        "law_ins": "\n以上文字是上市公司的简介，请基于金融知识，将其压缩为一个表示，可以与相关的金融政策做匹配，这个表示是：",  
    },
    "from_dialogmed_new": {
        "data_ins": "\n以上文字是患者自述的症状，请基于医疗知识，将其压缩为一个表示，用来检索相关的药物说明书，这个表示是：", 
        "law_ins": "\n以上文字是药物的说明书，请基于医疗知识，将其压缩为一个表示，可以用来与患者的症状做匹配，这个表示是：",  
    },
}

maps = {'evalins0': data_ins_map0, 'evalins1': data_ins_map1, 'evalins2': data_ins_map2}
max_length = 512

for evalins in [
    'evalins0', 
    # 'evalins1', 
    # 'evalins2'
]:
    data_ins_map = maps[evalins]

    for checkpoint in [
        "",
    ]:
        for (train_method, gist_token, use_ins) in [
            ("Baichuan2-7B-Chat", "#-以-上-内-容-的-总-结-#", True),
            ("Baichuan2-7B-Chat_manual_1K_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#", "#-以-上-内-容-的-总-结-#", True),
            ("Baichuan2-7B-Chat_manual_1K_ins_0.0_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#", "#-以-上-内-容-的-总-结-#", True),
            ("Baichuan2-7B-Chat_manual_1K_no_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#", "#-以-上-内-容-的-总-结-#", False),
            ("Baichuan2-7B-Chat_manual_1K_single_ins_0.2_2_1e-4_16_prompt0_#-以-上-内-容-的-总-结-#", "#-以-上-内-容-的-总-结-#", True),
        ]:
            
            
            if train_method == "Baichuan2-7B-Chat":
                checkpoint = "none"
            else:
                checkpoint = "final"
            
            method_start_time = time.time()
            print("\n\n========================================================================")
            
            if "Baichuan2-7B-Chat" in train_method:
                base_model_path = f'/ceph_home/XXX/Baichuan2-7B-Chat'
            elif "Baichuan2-7B-Base" in train_method:
                base_model_path = f'/ceph_home/XXX/Baichuan2-7B-Base'
            else:
                raise Exception(f"train_method error: {train_method}")
            tokenizer = AutoTokenizer.from_pretrained(f'{base_model_path}', use_fast=False, trust_remote_code=True)
            print(tokenizer)
            
            if checkpoint == "final":
                lora_path = f"/ceph_home/XXX/llm_matching/llm_matching_ins/{train_method}"
            elif checkpoint == "none":
                lora_path = None
            else:
                lora_path = f"/ceph_home/XXX/llm_matching/llm_matching_ins/{train_method}/checkpoint-{checkpoint}"
            print(lora_path)
            embedding_method = f'embedding_by_{train_method}_checkpoint-{checkpoint}'
            print("embedding_method:", embedding_method)

            model = AutoModelForCausalLM.from_pretrained(f'{base_model_path}', device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
            if lora_path is not None:
                print("will add lora")
                model = PeftModel.from_pretrained(
                    model,
                    lora_path,
                    torch_dtype=torch.float16,
                )
                print("lora added\n")
            model.eval()
            print(model.config)

            if gist_token != "no-gist":
                gist_input_ids = tokenizer(gist_token, return_tensors="pt").input_ids.to(model.device)
                gist_attention_mask = torch.ones(gist_input_ids.shape).to(model.device)
                print("gist_input_ids:", gist_input_ids)

            for data_name in [
                "policy-company",
                "case-provision", 
                "symptom-drug", 
                "objective-course"
            ]: # about 10 minutes for total three datasets
                print("\n\n=========================")
                print(data_name)

                laws = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws.json", 'r'))
                print(len(laws))
                print(laws[0].keys())
                datas = json.load(open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas.json", 'r'))
                print(len(datas))
                print(datas[0].keys())
                print(datas[0]['context'])

                s_time = time.time() # calculte the time as second for each 100 data
                for i, law in enumerate(laws):
                    if i % 100 == 0:
                        print(f"{i}/{len(laws)}, this 100 data cost {time.time() - s_time} seconds")
                        s_time = time.time()

                    inputs0 = tokenizer(law['law_content'], return_tensors="pt", max_length=max_length, truncation=True).to(model.device)
                    if use_ins:
                        inputs1 = tokenizer(data_ins_map[data_name]["law_ins"], return_tensors="pt").to(model.device)
                        inputs_ids = torch.cat([inputs0.input_ids, inputs1.input_ids], dim=-1)
                        inputs_attention_mask = torch.cat([inputs0.attention_mask, inputs1.attention_mask], dim=-1)
                    else:
                        inputs_ids = inputs0.input_ids
                        inputs_attention_mask = inputs0.attention_mask
                    if gist_token != "no-gist":
                        gist_start = inputs_ids.shape[1]
                        # print("gist_start:", gist_start)
                        total_input_ids = torch.cat([inputs_ids, gist_input_ids], dim=-1)
                        total_attention_mask = torch.cat([inputs_attention_mask, gist_attention_mask], dim=-1)
                    else:
                        gist_start = inputs_ids.shape[1] - 1 # the last token is the gist token
                        total_input_ids = inputs_ids
                        total_attention_mask = inputs_attention_mask
                    with torch.no_grad():            
                        outputs = model(input_ids=total_input_ids, attention_mask=total_attention_mask, output_hidden_states=True, return_dict=True)
                    last_hidden_states = outputs.hidden_states[-1] # 1*len*4096
                    # print("last_hidden_states.shape:", last_hidden_states.shape)
                    law_embedding = torch.mean(last_hidden_states[:, gist_start:, :], dim=1) # law_embedding.shape == (1, 768)
                    # print(law_embedding)
                    # input()
                    law_embedding = law_embedding.tolist()
                    law["embedding"] = law_embedding
                json.dump(laws, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/laws_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
                
                print('\nlaw embdding done\n')

                s_time = time.time() # calculte the time as second for each 100 data
                for i, data in enumerate(datas):
                    if i % 100 == 0:
                        print(f"{i}/{len(datas)}, this 100 data cost {time.time() - s_time} seconds")
                        s_time = time.time()
                    
                    inputs0 = tokenizer(data['context'], return_tensors="pt", max_length=max_length, truncation=True).to(model.device)
                    if use_ins:
                        inputs1 = tokenizer(data_ins_map[data_name]["data_ins"], return_tensors="pt").to(model.device)
                        inputs_ids = torch.cat([inputs0.input_ids, inputs1.input_ids], dim=-1)
                        inputs_attention_mask = torch.cat([inputs0.attention_mask, inputs1.attention_mask], dim=-1)
                    else:
                        inputs_ids = inputs0.input_ids
                        inputs_attention_mask = inputs0.attention_mask
                    if gist_token != "no-gist":
                        gist_start = inputs_ids.shape[1]
                        total_input_ids = torch.cat([inputs_ids, gist_input_ids], dim=-1)
                        total_attention_mask = torch.cat([inputs_attention_mask, gist_attention_mask], dim=-1)
                    else:
                        gist_start = inputs_ids.shape[1] - 1 # the last token is the gist token
                        total_input_ids = inputs_ids
                        total_attention_mask = inputs_attention_mask
                    with torch.no_grad():
                        outputs = model(input_ids=total_input_ids, attention_mask=total_attention_mask, output_hidden_states=True, return_dict=True)
                    last_hidden_states = outputs.hidden_states[-1] # 1*len*768
                    # print("last_hidden_states.shape:", last_hidden_states.shape)
                    data_embedding = torch.mean(last_hidden_states[:, gist_start:, :], dim=1) # data_embedding.shape == (1, 768)
                    data_embedding = data_embedding.tolist()
                    data["embedding"] = data_embedding
                json.dump(datas, open(f"/ceph_home/XXX/llm_matching/datas_clear/{data_name}/datas_{embedding_method}.json", 'w'), indent=4, ensure_ascii=False)
                
            print(f'{embedding_method} done, cost minutes: {(time.time() - method_start_time) / 60}') # 20 minute for 3 datasets total
            
            # print(torch.cuda.memory_summary())
            del model
            del laws
            del datas
            del law_embedding
            del data_embedding
            del last_hidden_states
            del outputs
            torch.cuda.empty_cache() # useful to solve too slow in the 3rd time run model
            # print(torch.cuda.memory_summary())
        
print("all done")