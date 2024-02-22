from transformers.trainer import *
import logging
logger = logging.getLogger(__name__)
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

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

class BiTrainer(Trainer):
    
    def __init__(self, save_lora_checkpoint, lora_bias, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_lora_checkpoint = save_lora_checkpoint
        self.lora_bias = lora_bias
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        elif self.save_lora_checkpoint:
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.lora_bias # the model is PETF model
            )
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # # save the checkpoint for sentence-transformers library
        # if self.is_world_process_zero():
        #     save_ckpt_for_sentence_transformers(output_dir,
        #                                         pooling_mode=self.args.sentence_pooling_method,
        #                                         normlized=self.args.normlized)

    def compute_loss(self, model, inputs, return_outputs=False, loss_type="gist"):
        
        outputs = model(**inputs, loss_type=loss_type)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        loss1 = None
        loss1r = None
        if self.args.loss_rate > 0: # need gist loss1
            with self.compute_loss_context_manager():
                loss1 = self.compute_loss(model, inputs, loss_type="gist")
            if self.args.n_gpu > 1:
                loss1 = loss1.mean()  # mean() to average on multi-gpu parallel training
            # logger.info("loss1: %s", loss1)
            loss1 *= (self.args.loss_rate/2)
            self.accelerator.backward(loss1)
            loss1 = loss1 / self.args.gradient_accumulation_steps
                
                
            with self.compute_loss_context_manager():
                loss1r = self.compute_loss(model, inputs, loss_type="gist-reverse")
            if self.args.n_gpu > 1:
                loss1r = loss1r.mean()  # mean() to average on multi-gpu parallel training
            # logger.info("loss1r: %s", loss1r)
            loss1r *= (self.args.loss_rate/2)
            self.accelerator.backward(loss1r)
            loss1r = loss1r / self.args.gradient_accumulation_steps


        loss2 = None
        loss2r = None
        if self.args.loss_rate < 1: # need contrastive loss2
            with self.compute_loss_context_manager():
                loss2 = self.compute_loss(model, inputs, loss_type="contrastive")
            if self.args.n_gpu > 1:
                loss2 = loss2.mean()  # mean() to average on multi-gpu parallel training
            # logger.info("loss2: %s", loss2)
            loss2 *= ((1-self.args.loss_rate)/2)
            self.accelerator.backward(loss2)
            loss2 = loss2 / self.args.gradient_accumulation_steps

                
            with self.compute_loss_context_manager():
                loss2r = self.compute_loss(model, inputs, loss_type="contrastive-reverse")
            if self.args.n_gpu > 1:
                loss2r = loss2r.mean()  # mean() to average on multi-gpu parallel training
            # logger.info("loss2r: %s", loss2r)
            loss2r *= ((1-self.args.loss_rate)/2)
            self.accelerator.backward(loss2r)
            loss2r = loss2r / self.args.gradient_accumulation_steps
                
                
        total_loss = 0
        # logger.info("loss1, loss1r, loss2, loss2r: %s, %s, %s, %s", loss1, loss1r, loss2, loss2r)
        if loss1 != None:
            total_loss += loss1.detach() 
        if loss1r != None:
            total_loss += loss1r.detach()
        if loss2 != None:
            total_loss += loss2.detach()
        if loss2r != None:
            total_loss += loss2r.detach()
        
        return total_loss

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     # if is_sagemaker_mp_enabled():
    #     #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #     #     return loss_mb.reduce_mean().detach().to(self.args.device)

    #     # #############################################################################################################
    #     # with self.compute_loss_context_manager():
    #     #     loss = self.compute_loss(model, inputs, loss_type="debug")
    #     # if self.args.n_gpu > 1:
    #     #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #     # if self.do_grad_scaling:
    #     #     self.scaler.scale(loss).backward()
    #     # elif self.use_apex:
    #     #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #     #         scaled_loss.backward()
    #     # else:
    #     #     self.accelerator.backward(loss)
    #     # total_loss =  loss.detach() / self.args.gradient_accumulation_steps
    #     # #############################################################################################################
        
    #     total_loss = 0
        
    #     if self.args.loss_rate > 0: # need gist loss
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, inputs, loss_type="gist")
    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #         loss *= (self.args.loss_rate/2)
    #         # logger.info("gist loss: %s", loss)
    #         if self.do_grad_scaling:
    #             self.scaler.scale(loss).backward()
    #         elif self.use_apex:
    #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             self.accelerator.backward(loss)
    #         total_loss += loss.detach() / self.args.gradient_accumulation_steps
            
    #         # with self.compute_loss_context_manager():
    #         #     loss = self.compute_loss(model, inputs, loss_type="gist-reverse")
    #         # if self.args.n_gpu > 1:
    #         #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #         # loss *= (self.args.loss_rate/2)
    #         # # logger.info("gist-reverse loss: %s", loss)
    #         # if self.do_grad_scaling:
    #         #     self.scaler.scale(loss).backward()
    #         # elif self.use_apex:
    #         #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #         #         scaled_loss.backward()
    #         # else:
    #         #     self.accelerator.backward(loss)
    #         # total_loss += loss.detach() / self.args.gradient_accumulation_steps
            
    #     if self.args.loss_rate < 1: # need contrastive loss
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, inputs, loss_type="contrastive")
    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #         loss *= ((1-self.args.loss_rate)/2)
    #         # logger.info("contrastive loss: %s", loss)
    #         # logger.info("self.do_grad_scaling %s", self.do_grad_scaling)
    #         if self.do_grad_scaling:
    #             self.scaler.scale(loss).backward()
    #         elif self.use_apex:
    #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             self.accelerator.backward(loss)
    #         total_loss += loss.detach() / self.args.gradient_accumulation_steps

    #         # with self.compute_loss_context_manager():
    #         #     loss = self.compute_loss(model, inputs, loss_type="contrastive-reverse")
    #         # if self.args.n_gpu > 1:
    #         #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #         # loss *= ((1-self.args.loss_rate)/2)
    #         # # logger.info("contrastive-reverse loss: %s", loss)
    #         # # logger.info("self.do_grad_scaling %s", self.do_grad_scaling)
    #         # if self.do_grad_scaling:
    #         #     self.scaler.scale(loss).backward()
    #         # elif self.use_apex:
    #         #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #         #         scaled_loss.backward()
    #         # else:
    #         #     self.accelerator.backward(loss)
    #         # total_loss += loss.detach() / self.args.gradient_accumulation_steps
            
    #     return total_loss