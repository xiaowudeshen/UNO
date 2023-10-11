# Copyright (c) Facebook, Inc. and its affiliates
# Code modified from the origianl T5DST work
from builtins import breakpoint
import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import random
from functools import partial
from utils.fix_label import fix_general_label_error
from collections import OrderedDict
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
BINARY_SLOT_LIST = ["hotel-parking", 'hotel-internet' ]
random.seed(577)
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)


# ## Code copy from summarization DS2, using the naive method for this step
# def state_to_sum(slot_values):
#     """
#     If we wants to generate various templates and lm ranking, we could fit better preposition for each slot
#     Input:
#         example: {'domain-key1': 'value1', 'key2': 'value2'}
#     Returns:
#         example: "The user wants key1 as value1, key2 as value2"
#         real_ex: "The user wants london as departure, cambridge as destination, 12:30 as arriveby, 3 as book people,
#                 tuesday as day."
#     """
#     sentence_prefix = 'The user wants '
#     res = sentence_prefix
#     for i, (domain_slot, value) in enumerate(slot_values.items()):
#         if i > 0:
#             res += ", "
#         sum_domain = domain_slot.split('-')[0]
#         sum_slot = domain_slot.split('-')[-1]
#         phrase = value + ' as ' + sum_slot + ' for ' +sum_domain
#         res += phrase
        

#     post_phrase = ' for the booking.'
#     res += post_phrase
#     return res

def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    choice_token = " <extra_id_0> "
    print(("Reading all files from {}".format(path_name)))
    data = []
    domain_counter = {}
    training_samples = {
        "main_task": 0,
        "value": 0,
        "none": 0,
        "aux_task_mask": 0,
        "aux_task_value": 0
    }
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        if dataset=="train" and args["fewshot"]>0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials)*args["fewshot"])]

        

        for dial_dict in dials:
            dialog_history = ""

            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue


            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_id = ti

                # accumulate dialogue utterances
                dialog_wo_system = dialog_history + " system: " + choice_token + " user: " + turn["user"]
                dialog_history +=  (" System: " + turn["system"] + " User: " + turn["user"])
                if args["fix_label"]:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"],SLOTS)
                else:
                    slot_values = turn["state"]["slot_values"]
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                
                if not dataset == 'test':
                    turn_belief_list0 = [str(k)+'-'+str(v) for k,v in slot_values.items()]
                    turn_belief_list = turn_belief_list0
                    if "slot_type" in args["joint_training"]:
                        turn_belief_list1 = [k["masked_type"] for k in turn["masked_state"]]
                        turn_belief_list += turn_belief_list1
                    # if "summary" in args["joint_training"]:
                    if "discriminator" in args["joint_training"]:
                        turn_belief_list2 = [str(k)+'-Yes' for k,_ in slot_values.items()]
                        turn_belief_list += turn_belief_list2

                elif dataset == 'test':
                    turn_belief_list = [str(k)+'-'+str(v) for k,v in slot_values.items()]


                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                if "gpt" in args["model_name"]:
                    turn_slots = []
                    turn_slot_values = []
                    if len(dialog_history.split())>800:
                        continue
                    for slot in slot_temp:
                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset !="test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}" + " " + tokenizer.bos_token
                        output_text = input_text+ " " + turn["state"]["slot_values"].get(slot, 'none').strip() + " " + tokenizer.eos_token
                        slot_text = slot
                        value_text = turn["state"]["slot_values"].get(slot, 'none').strip()

                        data_detail = {
                            "ID":dial_dict["dial_id"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief":turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text,
                            "question_type":"prediction"
                            }
                        data.append(data_detail)

                else:
                    for slot in slot_temp:

                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset !="test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue

                        output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                        slot_text = slot
                        value_text = slot_values.get(slot, 'none').strip()

                        if args["slot_lang"]=="human":
                            slot_lang = description[slot]["description_human"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="naive":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"]=="value":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"]=="question":
                            slot_lang = description[slot]["question"]
                            input_text = f"{dialog_history} {tokenizer.sep_token} {slot_lang}".lower()
                        elif args["slot_lang"]=="slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = f"{dialog_history} {tokenizer.sep_token} {slot_lang}".lower()
                        else:
                            input_text = f"{dialog_history} {tokenizer.sep_token} {slot}".lower()

                        data_detail = {
                            "ID":dial_dict["dial_id"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief":turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text,
                            "value_list":description[slot]["values"],
                            "question_type":"prediction"
                            }
                        if slot in slot_values:
                            training_samples["value"] += 1
                        else: 
                            training_samples["none"] += 1
                        training_samples["main_task"] += 1
                        
                        data.append(data_detail)
                        # breakpoint()
                        if dataset == 'train' and "value_slot" in args["joint_training"]:    
                            if args["slot_lang"]=="question":
                                slot_p = f"What is the slot type of the slot value {value_text}"
                            elif args["slot_lang"]=="slottype":
                                slot_p = f"slot type of the value {value_text}"
                            # breakpoint()
                            if slot in turn["state"]["turn_slot_values"].keys():
                                input_text = f"{dialog_history} {tokenizer.sep_token} {slot_p}?".lower()
                                output_text = slot_text + f" {tokenizer.eos_token}"

                                data_detail = {
                                    "ID":dial_dict["dial_id"],
                                    "domains":dial_dict["domains"],
                                    "turn_id":turn_id,
                                    "dialog_history":dialog_history,
                                    "turn_belief":turn_belief_list,
                                    "intput_text":input_text,
                                    "output_text":output_text,
                                    "slot_text":slot_text,
                                    "value_text":value_text,
                                    "value_list":description[slot]["values"],
                                    "question_type":"prediction"
                                    
                                    }
                                training_samples["aux_task_value"] += 1
                                
                                data.append(data_detail)   
                        


                        
                    # joining training with masked slot type included    
                    if dataset == 'train' and "mask_slot" in args["joint_training"]:   
                        if args["slot_lang"]=="question":
                            slot_p = 'What is the slot type of the masked token'
                        elif args["slot_lang"]=="slottype":
                            slot_p = 'masked slot type'
                        for masked_turn in turn['masked_state']:
                            context_mask = masked_turn["dialog_history"]
                            input_text = f"{context_mask} {tokenizer.sep_token} {slot_p}?".lower()
                            output_text = masked_turn['masked_type'] + f" {tokenizer.eos_token}"

                            data_detail = {
                                "ID":dial_dict["dial_id"],
                                "domains":dial_dict["domains"],
                                "turn_id":turn_id,
                                "dialog_history":dialog_history,
                                "turn_belief":turn_belief_list,
                                "intput_text":input_text,
                                "output_text":output_text,
                                "slot_text":slot_text,
                                "value_text":value_text,
                                "value_list":description[slot]["values"],
                                "question_type":"prediction"
                                
                                }
                            training_samples["aux_task_mask"] += 1
                            data.append(data_detail)

                   
                        
    # print(len(data))
    for idx in range(10):
        print(data[idx])
        # breakpoint()
    
    print("domain_counter", domain_counter)
    print("training_samples", training_samples)
    return data, slot_temp



def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS


def gpt_collate_fn(data,tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False, truncation=True, max_length=1000)
    batch_data["input_ids"] = output_batch['input_ids']
    return batch_data


def collate_fn(data, tokenizer, args):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["intput_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    # print(input_batch["input_ids"].size())
    # breakpoint()
    # if not args["turn_att"] == 'none':
        # Att = Create_Att_Matrix(input_batch["input_ids"], max_dist= args["max_dist"])
        # batch_data["turn_att_a"], batch_data["turn_att_b"] = convert2leng(input_batch["input_ids"])
    # print(batch_data["turn_att"].size())
    # batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, truncation = True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


def prepare_data(args, tokenizer):
    # path_train = 'data/train_dials.json'
    path_train = 'data1/mask_train_dials.json'
    # path_train = 'data1/ann_mask_train_dials.json' 
    # path_dev = 'data/dev_dials.json'
    path_dev = 'data1/mask_dev_dials.json'
    path_test = 'data/test_dials.json'

    ontology = json.load(open("data/MULTIWOZ2.1/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("utils/slot_description.json", 'r'))

    data_train, _ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, "train")
    data_dev, _ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
    data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")

    # data_train = data_train[:50]
    # data_dev = data_dev[:200]
    # data_test = data_test[:200]


    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    if "gpt" in args["model_name"]:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args), num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args), num_workers=2)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args), num_workers=2)
    fewshot_loader_dev=None
    fewshot_loader_test=None
    return train_loader, dev_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test
