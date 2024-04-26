from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel


@dataclass
class ScriptArguments:
    """
    Additional arguments for training, which are not part of TrainingArguments.
    """
    # model_id: str = field(
    #   metadata={
    #         "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
    #     },
    # )
    # dataset_path: Optional[str] = field(
    #     default="timdettmers/openassistant-guanaco",
    #     metadata={"help": "The preference dataset to use."},
    # )
    load_checkpoint: Optional[bool] = field(default=False)
    checkpoint_path: Optional[str] = field(default="")
    plt: Optional[bool] = field(default=False)
    plot_train: Optional[bool] = field(default=False)
    test: Optional[bool] = field(default=False)
    fuse: Optional[bool] = field(default=True)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    cutoff_len: Optional[int] = field(default=512)
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )


from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control


def load_dataset(args, train_file, doc_file, tokenizer):
    import json
    def read_data(file):
        with open(file, 'r') as fp:
            data = json.load(fp)
        return data  

    def generate_prompt(data, doc_data):
        # 直接拼成[INST]的形式 也可以
        # sys_msg= "Translate the given text to Shakespearean style."
        sys_msg = "You are a helpful bot who reads texts and answers questions about them."
        # print(data)
        tree_id = data["tree_id"]
        cur_doc = doc_data[tree_id]
        # print(cur_doc)
        # if cur_doc 
        edu_text = ''
        for edus in (cur_doc["edus"] if not cur_doc["is_bullet"] else cur_doc["clauses"]):
            if type(edus) == list:
                for edu in edus:
                    if edu_text == '':
                        edu_text = edu
                        continue
                    edu_text = edu_text + '\t' + edu
            else:
                if edu_text == '':
                    edu_text = edus
                    continue
                edu_text = edu_text + '\t' + edus
                
        history = ''
        for his in data["history"]:
            # print(his)
            if not history:
                history = his["follow_up_question"] + ' ' + his["follow_up_answer"]
            history =  history + '\t' + his["follow_up_question"] + ' ' + his["follow_up_answer"]
        if len(history) == 0:
            history = 'Empty'
        scenario = data["scenario"]
        if len(scenario) == 0:
            scenario = 'Empty'
        question = data["question"]
        answer = data["answer"]
        if args.test:
            if args.fuse:
                prompt = ("[INST]" + sys_msg + "Document: ", 
                      edu_text.split("\t"),
                      "Scenario: " + scenario + "Dialogue History: " + history + "Question: " + question + "[/INST] " + "Answer: ")
            else:
                prompt = "[INST]" + sys_msg + "Document: " + edu_text + "Scenario: " + scenario + "Dialogue History: " + history + "Question: " + question + "[/INST] " + "Answer: "
        elif args.fuse:
            prompt = ("[INST]" + sys_msg + "Document: ", 
                      edu_text.split("\t"),
                      "Scenario: " + scenario + "Dialogue History: " + history + "Question: " + question + "[/INST] " + "Answer:" + answer)
        else:
            prompt = "[INST]" + sys_msg + "Document: " + edu_text + "Scenario: " + scenario + "Dialogue History: " + history + "Question: " + question + "[/INST] " + " Answer:" + answer
        # print(prompt)
        # exit()
        return prompt
    
    
    def tokenize(args, tokenizer, prompt):
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=args.cutoff_len ,
            padding="max_length"
        )
    
    def tokenize_test(args, tokenizer, prompt):
        return tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len ,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
    
    def tokenize_plt(args, tokenizer, prompt):
        return tokenizer(prompt)
    
    def tokenize_token(tokenizer, token):
        return tokenizer(token, return_tensors="pt").to("cuda")
    
    def concat(data, new):
        data["input_ids"].extend(new["input_ids"][1:])
        data["attention_mask"].extend(new["attention_mask"][1:])
        return data
        
    def padding(tokenzier, data):
        pad_token = tokenizer.pad_token
        padding = [tokenizer(pad_token)["input_ids"][1]] * (args.cutoff_len - len(data["input_ids"]))
        mask_pad = [0]* len(padding)
        data["input_ids"] = padding + data["input_ids"]
        data["attention_mask"] = mask_pad + data["attention_mask"]
        return data
    doc_data = read_data(doc_file)
    train_ori_data = read_data(train_file)
    from tqdm import tqdm 
    if args.test and not args.fuse:
        print("using test")
        train_data = [tokenize_test(args, tokenizer, generate_prompt(x, doc_data)) for x in tqdm(train_ori_data)]
    elif args.plt:
        train_data = [tokenize_plt(args, tokenizer, generate_prompt(x, doc_data)) for x in tqdm(train_ori_data)]
    elif args.fuse:
        train_data = []
        for x in tqdm(train_ori_data):
            # print(x)
            prompt = generate_prompt(x, doc_data)
            # print(prompt)

            pre_ids = tokenize_token(tokenizer, prompt[0])
            input_ids = pre_ids["input_ids"]
            attention_mask = pre_ids["attention_mask"]
            edu_ids = torch.full(pre_ids["input_ids"].shape, -1, dtype=torch.int)

            # [1, 22, 4096]
            edu_indices = doc_data[x["tree_id"]]["enity_ids"]
            for idx, edu in enumerate(prompt[1]):
                edu_input_ids = tokenize_token(tokenizer, edu)
                edu_index = int(edu_indices[idx])
                input_ids = torch.cat(
                    [input_ids, edu_input_ids["input_ids"]], dim=1
                )
                attention_mask = torch.cat(
                    [attention_mask, edu_input_ids["attention_mask"]], dim=1
                )
                edu_ids = torch.cat(
                    [edu_ids, torch.full(edu_input_ids["input_ids"].shape, edu_index, dtype=torch.int)], dim=1
                )
            
            post_ids = tokenize_token(tokenizer, prompt[-1])
            input_ids = torch.cat(
                [input_ids, post_ids["input_ids"]], dim=1
            )
            attention_mask = torch.cat(
                [attention_mask, post_ids["attention_mask"]], dim=1
            )
            edu_ids = torch.cat(
                [edu_ids, torch.full(post_ids["input_ids"].shape, -1, dtype=torch.int)], dim=1
            )
            # print(edu_ids)

            # cur_train_data = tokenize_token(tokenizer, prompt[0])
            # for word in prompt[1:]: 
            #     cur_train_data = concat(cur_train_data, tokenize_token(tokenizer, word))
            # cur_train_data = concat(cur_train_data, tokenize_token(tokenizer, tokenizer.eos_token))
            # cur_train_data = padding(tokenizer, cur_train_data)
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(edu_ids.shape)
            if args.test:
                if args.fuse:
                    train_data.append({
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "edu_ids": edu_ids[0]
                })
                else:
                    train_data.append({
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0]
                })   
            else:
                train_data.append({
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "edu_ids": edu_ids[0]
                })
            
    else:
        train_data = [tokenize(args, tokenizer, generate_prompt(x, doc_data)) for x in tqdm(train_ori_data)]
    # print(train_data[0])
    # exit()
    return train_data

def print_trainable_parameters(m):
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

def main():
    # from main import ScriptArguments
    from transformers import HfArgumentParser, TrainingArguments
    parser = HfArgumentParser([ScriptArguments,TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    from typing import cast
    args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)
    print(args.fuse)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./Mistral-7B-instruct")
    tokenizer.pad_token = "!"
    
    train_file = "./data/sharc_raw/json/sharc_train.json"
    # train_file = "./data/sharc_raw/json/sharc_train_question_fixed.json"
    # train_file = "./data/sharc_raw/json/train_test.json"
    # train_doc_file = "./data/train_snippet_parsed.json"
    train_doc_file = "./RotatE_sharc_4/parsed/train_snippet_parsed_with_id.json"

    train_data = load_dataset(args, train_file, train_doc_file, tokenizer)

    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from modeling_mistral import MistralForCausalLM
    import torch
    model = MistralForCausalLM.from_pretrained("/home/rjm/mine/Mistral-7B-instruct", torch_dtype=torch.bfloat16)
    # model = MistralForCausalLM.from_pretrained("mistral-fuse-new-load-4epoch-qv-lora-save/checkpoint-300", torch_dtype=torch.float16)
    # model = MistralForCausalLM.from_pretrained("/home/rjm/mine/Mistral-7B-instruct") 
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # target_modules=[ "w1", "w2", "w3"],  #Only Training the "expert" layers
        target_modules=[
            "q_proj",     
            "v_proj",
            "up_proj",
            "down_proj",
        ],
        # target_modules=["query_key_value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # if args.load_checkpoint:
    #     trainer = Trainer.from_pretrained(args.checkpoint_path)
    # Create trainer and add callbacks
    model.config.use_cache = False
    trainer.accelerator.print(f"{trainer.model}")
            
    # trainer.model.print_trainable_parameters()
    trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
    
    if args.load_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model on main process
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()

    # TODO: add merge adapters
    # Save everything else on main process
    if trainer.args.process_index == 0:
        if script_args.merge_adapters:
            # merge adapter weights with base model and save
            # save int 4 model
            trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
            # clear memory
            del model
            del trainer
            torch.cuda.empty_cache()

            from peft import AutoPeftModelForCausalLM

            # load PEFT model in fp16
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )  
            # Merge LoRA and base model and save
            model = model.merge_and_unload()        
            model.save_pretrained(
                training_args.output_dir, safe_serialization=True, max_shard_size="8GB"
            )
        else:
            trainer.model.save_pretrained(
                training_args.output_dir, safe_serialization=True
            )

        # save tokenizer 
        tokenizer.save_pretrained(training_args.output_dir)

# from transformers import MistralPreTrainedModel, get_scheduler, MistralModel
# class MistralForFuseEmbedding(MistralPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = MistralModel(config)
#         self.normalize = True

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ):
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         transformer_outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         last_hidden_state = transformer_outputs.last_hidden_state
#         embeddings = self.last_token_pool(last_hidden_state, attention_mask)
#         if self.normalize:
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

#         return embeddings

#     def last_token_pool(self, last_hidden_states, attention_mask):
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def read_Graph_embedding():
    entity_file = "./RotatE_sharc_4/entity_embedding.npy"
    import numpy as np
    data = np.load(entity_file)
    print(data[:10])
    print(type(data), data.shape)
    return data


def read_entity():
    entity_file = "./RotatE_sharc_4/sharc/entities.dict"
    entity_dict = {}
    with open(entity_file, 'r') as fp:
        for line in fp:
            eid, entity = line.strip('\n').split('\t')
            #print(eid, entity)
            entity_dict.update({entity:int(eid)})
            #print(entity_dict)
            #exit()
    print(len(entity_dict))
    return entity_dict


def change_doc_file(entity_dict):
    # doc_file = "./RotatE_sharc_4/parsed/train_snippet_parsed.json"
    doc_file = "./RotatE_sharc_4/parsed/dev_snippet_parsed.json"
    import json
    with open(doc_file, 'r') as fp:
        data = json.load(fp)
    from tqdm import tqdm
    for key, value in tqdm(data.items()):
        value.update({"enity_ids":[]})
        cur_id = key
        if value["is_bullet"] == False:
            edus = value["edus"] 
            for edu in edus:
                for e in edu:
                    ori = e
                    # print(e)
                    e = "_".join(e.strip().split(" "))
                    # e = e.replace(" ", "_")
                    try:
                        entity_id = entity_dict[e]
                    except KeyError:
                        print("1after:",e)
                        print("1ori:",ori)
                        print(key)
                        entity_id = -1
                    value["enity_ids"].append(entity_id)
        elif len(value["clauses"]) >= 1:
            edus = value["clauses"]
            for e in edus:
                # print(e)
                ori = e
                if e[0] == '*':
                    if len(e) > 1:
                        e = e[2:].replace("\n","").replace("*","").replace(" ", "_").strip("\n")
                else:
                    e = e.replace("\n","").replace("*","").replace(" ", "_")
                # print(e)
                # exit()
                try:
                    entity_id = entity_dict[e]
                except KeyError:
                    print("2after:",e)
                    print("2ori:",ori)
                    print(key)
                    entity_id = -1
                value["enity_ids"].append(entity_id)
    # save_file = "./RotatE_sharc_4/parsed/train_snippet_parsed_with_id.json"
    save_file = "./RotatE_sharc_4/parsed/dev_snippet_parsed_with_id.json"
    with open(save_file, 'w') as fp:
        json.dump(data, fp, indent=2)
    

def predict():
    from transformers import HfArgumentParser, TrainingArguments
    parser = HfArgumentParser([ScriptArguments,TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    from typing import cast
    args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)
    print(args.test, args.fuse)
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import deepspeed
    import torch
    model = AutoPeftModelForCausalLM.from_pretrained("./mistral-fuse-new-load-4epoch-qv-lora-save/checkpoint-3600", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("./mistral-fuse-new-load-4epoch-qv-lora-save/checkpoint-3600")
    tokenizer.pad_token = "!"
    # model = AutoModelForCausalLM.from_pretrained("./Mistral-7B-instruct")
    # tokenizer = AutoTokenizer.from_pretrained("./Mistral-7B-instruct")
    # dev_file = "./data/sharc_raw/json/sharc_dev.json"
    # dev_doc_file = "./data/dev_snippet_parsed.json"
    # dev_data = load_dataset(args, dev_file, dev_doc_file, tokenizer)
    ds_model = deepspeed.init_inference(
        model=model,      # Transformers模型
        mp_size=1,        # GPU数量
        dtype=torch.float16, # 权重类型(fp16)
        replace_method="auto", # 让DS自动替换层
        replace_with_kernel_inject=True, # 使用kernel injector替换
    )
    print(f"模型加载至设备{ds_model.module.device}\n")
    from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
    # assert isinstance(ds_model.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"

    dev_file = "./data/sharc_raw/json/sharc_dev.json"
    # dev_doc_file = "./data/dev_snippet_parsed.json"
    dev_doc_file = "./RotatE_sharc_4/parsed/dev_snippet_parsed_with_id.json"
    dev_data = load_dataset(args, dev_file, dev_doc_file, tokenizer)
    # def gen_after_tunning(model, dev_data, maxlen=1024, sample=True):
    results = []
    # ds_model.eval()
    model.eval()
    from tqdm import tqdm
    count = 0
    with torch.no_grad():
        for data in tqdm(dev_data):
            if args.fuse:
                # print(data)
                # print(data["input_ids"].shape)
                # input = torch.unsqueeze(data["input_ids"], 0)
                # print(input.shape)
                # print(**data)
                out = model.generate(**data, max_new_tokens=256, num_return_sequences=1)
            else:
                out = ds_model.generate(**data, max_new_tokens=256, num_return_sequences=1)
            res = tokenizer.batch_decode(out[0], skip_special_tokens=True)
            save_file = "./data/fuse_4eporch_3600_devfalsefuse_results.json"
            import json
            
            res = ''.join(res)
            results.append(res)
            with open(save_file, 'w') as fp:
                for res in results:
                    # print(res)
                    fp.write(res + '\n') 
        
if __name__=='__main__':
    # main()
    # read_Graph_embedding()
    # entity_dict = read_entity()
    # change_doc_file(entity_dict)
    predict()
    