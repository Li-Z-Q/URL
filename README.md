# URL: Universal Referential Knowledge Linking via Task-instructed Representation Compression

### Catalogue
```bash
url                         
├─ training_data                                     
│  ├─ manual_1K_ins.jsonl                            
│  ├─ manual_1K_ins_en.jsonl                         
│  ├─ manual_1K_no_ins.jsonl                         
│  ├─ manual_1K_no_ins_en.jsonl                      
│  ├─ manual_1K_single_ins.jsonl                     
│  └─ manual_1K_single_ins_en.jsonl                  
├─ urlbench                                          
│  ├─ case-provision                                 
│  │  ├─ datas.json                                 
│  │  └─ laws.json                                   
│  ├─ case-provision_en                              
│  │  ├─ datas.json                                  
│  │  └─ laws.json                                   
│  ├─ objective-course                               
│  │  ├─ datas.json                                  
│  │  └─ laws.json                                   
│  ├─ objective-course_en                            
│  │  ├─ datas.json                                  
│  │  └─ laws.json                                   
│  ├─ policy-company                                 
│  │  ├─ datas.json                                  
│  │  └─ laws.json                                   
│  ├─ policy-company_en                              
│  │  ├─ datas.json                                  
│  │  └─ laws.json                                   
│  ├─ symptom-drug                                   
│  │  ├─ datas.json                                  
│  │  └─ laws.json                                   
│  └─ symptom-drug_en                                
│     ├─ datas.json                                  
│     └─ laws.json                                   
├─ utils                                             
│  ├─ arguments.py                                   
│  ├─ data_ins.py                                    
│  ├─ modeling_baichuan.py                           
│  ├─ modeling_ins.py                                
│  ├─ modeling_ins_en.py                             
│  ├─ modeling_llama.py                              
│  ├─ peft_model.py                                  
│  ├─ save_and_load.py                               
│  ├─ SentenceTransformer.py                         
│  ├─ trainer_ins.py                                 
│  └─ trainer_ins_en.py                              
├─ generate_embedding_by_baichuan.py                 
├─ generate_embedding_by_baichuan_en.py              
├─ generate_embedding_by_baichuan_with_eval_ins.py   
├─ generate_embedding_by_e5.py                       
├─ generate_embedding_by_e5_en.py                    
├─ generate_embedding_by_encoder.py                  
├─ generate_embedding_by_encoder_en.py               
├─ generate_embedding_by_encoder_trained.py          
├─ generate_embedding_by_encoder_trained_en.py       
├─ generate_embedding_by_instructor_en.py            
├─ generate_embedding_by_llm_lora_train.py           
├─ generate_embedding_by_llm_lora_train_en.py        
├─ generate_embedding_by_openai.py                   
├─ generate_embedding_by_openai_en.py                
├─ generate_embedding_by_openai_with_eval_ins.py     
├─ generate_embedding_by_openai_with_eval_ins_en.py  
├─ generate_embedding_by_uae.py                      
├─ generate_embedding_by_voyage_en.py                
├─ generate_embedding_by_voyage_with_eval_ins.py     
├─ match_by_embedding_sim.py                         
├─ match_by_embedding_sim_en.py                      
├─ run_ins.py                                        
├─ run_ins.sh                                        
├─ run_ins_en.py                                     
├─ run_ins_en.sh                                     
├─ show_results_new.py                               
└─ show_results_new_en.py  
```

### Prepare environment
```bash
conda create -n url python=3.8
conda activate url
pip install -r requirement.txt
```

### Evaluation
#### Prepare data
```bash
# */datas.json is Claim database
# */laws.json is Reference database
cp -r ./urlbench/* /ceph_home/XXX/llm_matching/datas_clear/
```

#### Generate embeddings
```bash
# For Chinese evaluation
python generate_embedding_by_llm_lora_train.py # ours, lora weight is in another .zip file
python generate_embedding_by_baichuan.py                 
python generate_embedding_by_baichuan_with_eval_ins.py   
python generate_embedding_by_e5.py                       
python generate_embedding_by_encoder.py                  
python generate_embedding_by_encoder_trained.py          
python generate_embedding_by_openai.py                   
python generate_embedding_by_openai_with_eval_ins.py     

# For English evaluation
python generate_embedding_by_llm_lora_train_en.py # ours, lora weight is in another .zip file      
python generate_embedding_by_baichuan_en.py              
python generate_embedding_by_e5_en.py                    
python generate_embedding_by_encoder_en.py               
python generate_embedding_by_encoder_trained_en.py       
python generate_embedding_by_instructor_en.py            
python generate_embedding_by_openai_en.py                
python generate_embedding_by_openai_with_eval_ins_en.py  
python generate_embedding_by_uae.py                      
python generate_embedding_by_voyage_en.py                
python generate_embedding_by_voyage_with_eval_ins.py
```

#### Calculate similarity
```bash
# For Chinese evaluation
python match_by_embedding_sim.py
# For English evaluation
python match_by_embedding_sim_en.py
```

#### Show results
```bash
# For Chinese evaluation
python show_results_new.py
# For English evaluation
python show_results_new_en.py
```

### Training URL

##### Prepare data
```bash
cp ./training_data/* /ceph_home/XXX/llm_matching/
```

##### Train model
```bash
# For Chinese evaluation
bash run_ins.sh
# For English evaluation
bash run_ins_en.sh
```