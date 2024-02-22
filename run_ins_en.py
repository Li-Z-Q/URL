import logging
import os
from pathlib import Path
import random
random.seed(42)
import torch
torch.manual_seed(42)

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from utils.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from utils.data_ins import TrainDatasetForEmbedding, EmbedCollator
from utils.modeling_ins_en import BiEncoderModel
from utils.trainer_ins_en import BiTrainer

logger = logging.getLogger(__name__)

######################################################################### lora
import typing
from dataclasses import dataclass, field
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return
######################################################################### lora


def main():
    # parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    ######################################################################## lora
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    lora_args: LoraArguments
    ######################################################################## lora
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        trust_remote_code=True, 
        padding_side="left" # because use the last token as gist token
    )
    # here is very weird, i do not know why pad token will affect learn rate and performance
    # 1. here Baichun is a conincident error, i originally want it is Baichuan
    # 2. then i found that baichuan model's pad_token must be eos_token, bos and unk is all bad, will cause learnrate keep 0 in first several steps
    # 3. there is no time to find out the real reason, just keep it
    if "Baichun" not in model_args.model_name_or_path:
        # tokenizer.pad_token = tokenizer.eos_token # if using eos_token, llama's loss will be nan
        tokenizer.pad_token = tokenizer.bos_token
    logger.info('Tokenizer: %s', tokenizer)
    test_sentences = ["who are you", "i am"]
    test_inputs = tokenizer(test_sentences, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)
    logger.info('test_inputs: %s', test_inputs)
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=model_args.cache_dir,
    #     trust_remote_code=True
    # )
    # logger.info('Config: %s', config)

    # if training_args.loss_rate == 0:
    #     assert training_args.gist_token == "no-gist" # no need, contrasstive can also use gist_token
    if training_args.gist_token == "no-gist":
        gist_token_ids = None
    else:
        gist_token_ids = tokenizer([training_args.gist_token], add_special_tokens=False, return_tensors='pt').input_ids
    if training_args.prompt == "instance-specific":
        prompt_ids = None
    else:
        prompt_ids = tokenizer([training_args.prompt], add_special_tokens=False, return_tensors='pt').input_ids
    # print("prompt_ids", prompt_ids)
    # print("gist_token_ids", gist_token_ids)
    logger.info("prompt_ids: %s", prompt_ids)
    logger.info("gist_token_ids: %s", gist_token_ids)
    
    logger.info('Loading model...')
    assert training_args.use_data_ins == True, f"should keep {training_args.use_data_ins} True, because will contral ins by dataset"
    model = BiEncoderModel(
        model_name=model_args.model_name_or_path,                        
        normlized=training_args.normlized,
        # sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        prompt_ids=prompt_ids,
        loss_rate=training_args.loss_rate,
        gist_token_ids=gist_token_ids, 
        use_data_ins=training_args.use_data_ins,
    )
    logger.info('cuda memory allocated: %d', torch.cuda.memory_allocated() / 1024 / 1024)
    logger.info('load model done, will add lora')
    
    ######################################################################## lora
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        # task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info('cuda memory allocated after lora: %d', torch.cuda.memory_allocated() / 1024 / 1024)
    ######################################################################## lora

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer, 
        save_lora_checkpoint=True,
        lora_bias=lora_args.lora_bias
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    # trainer.save_model()
    ######################################################################## lora
    # # check if zero3 mode enabled
    # if deepspeed.is_deepspeed_zero3_enabled():
    #     # use deepspeed engine internal function to gather state dict
    #     # state_dict_zero3 contains whole parameters of base and lora adapters
    #     # we will not extract lora parameters since peft save_pretrained will do that
    #     # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
    #     # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
    #     state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    #     if training_args.local_rank == 0:
    #         state_dict = state_dict_zero3
    # else:
    #     # in other mode we use original code from fastchat team, to make sure our change is minimum
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), lora_args.lora_bias
    )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    ######################################################################## lora
    
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
