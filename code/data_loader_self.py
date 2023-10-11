# Copyright (c) Facebook, Inc. and its affiliates
# Code modified from the origianl T5DST work
from builtins import breakpoint
import json
from tkinter import N
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import numpy as np
import os
import random
from random import randrange
from functools import partial
from utils.fix_label import fix_general_label_error
from collections import OrderedDict
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
# 

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


def negative_sampling(slot, value, num):
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    answer_list = ontology[slot]
    for _ in range(num):
        i = randrange(len(answer_list))
        item = answer_list[i]   
    
    return item
    
def get_masked_input_and_labels(encoded_texts):
    breakpoint()
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]
    

def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    choice_token = " <extra_id_0> "
    print(("Reading all files from {}".format(path_name)))
    # if 'dev' in path_name:
    #     breakpoint()
    if args["self_training"] == "R3" and dataset == 'train':
        f = open(f"data_self/Selected_w_dialogs_" + args["only_domain"] +".json")
        good_labels = json.load(f)
        good_label_list = good_labels["label_list"]
        ##test only 50% of the good labels
        # short_good_list = random.sample(good_label_list, len(good_label_list)%2)
        # good_label_list = short_good_list
    data = []
    domain_counter = {}
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        if dataset=="train" and args["fewshot"]>0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials)*args["fewshot"])]

        for dial_dict in dials:
            dialog_history = ""
            if args["self_training"] == 'R3' and dataset == 'train' and  dial_dict["dial_id"] not in good_label_list:
                
                continue
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
                
                if args["self_training"] == "R2":
                    turn_belief_list = [str(k) for k, _ in slot_values.items()]
                else:
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
                            "question_type": "prediction"
                            }
                        data.append(data_detail)

                else:
                    for slot in slot_temp:
                        # if args["self_training"] == 'R2':
                        #     continue
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
                            # input_text = f"{dialog_history} {tokenizer.sep_token} {slot_lang}".lower()
                            input_text = f"extractive question: {slot_lang}? context: {dialog_history}".lower()
                        elif args["slot_lang"]=="slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = f"{dialog_history} {tokenizer.sep_token} {slot_lang}".lower()
                        else:
                            input_text = f"{dialog_history} {tokenizer.sep_token} {slot}".lower()

                        
                        output_text = value_text + f" {tokenizer.eos_token}"
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
                                "question_type": "prediction"
                                
                                }
                        if not args["self_training"] == "R2":
                            data.append(data_detail)
                        elif dataset != 'test':
                            data.append(data_detail)
                        
                           
                            
                    if args["self_training"] == "R2" and dataset == 'test':
                        slot_p = 'What is the slot type of the masked token'

                        
                        for masked_turn in turn['masked_state']:
                            # if 'value' in args["joint_training"]:
                            #     value_text = masked_turn["slot_value"]
                            #     slot_p = f"What is the slot type of the slot value {value_text}"
                            context_mask = masked_turn["dialog_history"]
                            input_text = f"extractive question: {slot_p}? context: {context_mask}".lower()
                            # input_text = f"{context_mask} {tokenizer.sep_token} {slot_p}".lower()
                            # input_text = f"extractive question: {slot_p}? context: {context_mask}".lower()
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
                                "question_type": "prediction"
                                }
                            
                            data.append(data_detail)

                    if args["self_training"] == "R3" and dataset == 'train' and "augmented" in path_name:
                        

                        
                        for masked_turn in turn['masked_state']:
                            if args["slot_lang"]=="question":
                                slot_p = description[masked_turn["masked_type"]]["question"]
                            elif args["slot_lang"]=="slottype":
                                slot_p = description[masked_turn["masked_type"]]["slottype"]
                            context_mask = masked_turn["dialog_history"]
                            # input_text = f"{context_mask} {tokenizer.sep_token} {slot_p}".lower()
                            input_text = f"extractive question: {slot_p}? context: {context_mask}".lower()
                            output_text = masked_turn['slot_value'] + f" {tokenizer.eos_token}"

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
                                "value_list":description[masked_turn["masked_type"]]["values"],
                                "question_type": "prediction"
                                }
                            
                            data.append(data_detail)


                    # joining training with slot type included    
                    # if args["self_training"] == "R3" and dataset == 'train' and "slot" in args["joint_training"]:    
                    #     slot_p = 'What is the slot type of the masked token'
                    #     for masked_turn in turn['masked_state']:
                            
                    #         context_mask = masked_turn["dialog_history"]
                    #         input_text = f"extractive question: {slot_p}? context: {context_mask}".lower()
                    #         output_text = masked_turn['masked_type'] + f" {tokenizer.eos_token}"

                    #         data_detail = {
                    #             "ID":dial_dict["dial_id"],
                    #             "domains":dial_dict["domains"],
                    #             "turn_id":turn_id,
                    #             "dialog_history":dialog_history,
                    #             "turn_belief":turn_belief_list,
                    #             "intput_text":input_text,
                    #             "output_text":output_text,
                    #             "slot_text":slot_text,
                    #             "value_text":value_text,
                    #             "value_list":description[slot]["values"],
                    #             "question_type": "extractive"
                    #             }
                            
                    #         data.append(data_detail)                    
    # print(len(data))
    if len(data) > 10:
        for idx in range(10):
            print(data[idx])
        print("domain_counter", domain_counter)
    else:
        print(data)
        print("domain_counter", domain_counter)


    # print(data)
    
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
    output_batch = tokenizer(batch_data["output_text"], padding=True, truncation = True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


def prepare_data(args, tokenizer):

    if args["self_training"] == 'R1':
        path_train = 'data/train_dials.json'
        path_dev = 'data1/mask_dev_dials.json'
        path_test = 'data/train_dials.json'
    elif args["self_training"] == 'R2':
        path_train = 'data/train_dials.json'
        path_dev = 'data1/mask_dev_dials.json'
        path_test = 'data_self/slot_train_dials_' + args["only_domain"] + '.json'
    elif args["self_training"] == 'R3':
        path_train = 'data_self/slot_train_dials_' + args["only_domain"] + '.json'
        # path_train = 'data_self/augmented_dialogues_' +args["only_domain"] + '.json'
        path_dev = 'data1/mask_dev_dials.json'
        path_test = 'data/test_dials.json'



    ontology = json.load(open("data/MULTIWOZ2.1/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("utils/slot_description.json", 'r'))

    data_train, _ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, "train")
    data_dev, _ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
    data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")

    # data_train = data_train[:200]
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
