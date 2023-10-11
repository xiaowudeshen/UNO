# Copyright (c) Facebook, Inc. and its affiliates
# Code modified from the origianl T5DST work
from builtins import breakpoint
import os, random
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
# from data_loader import prepare_data
from data_loader_self import prepare_data
from config import get_args
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from collections import Counter
import wandb
# def consistency_cross_entropy(lm_logits1, lm_logits2, threshold=0.4):
#     logsoftmax = torch.nn.LogSoftmax(dim=1)
#     softmax = torch.nn.Softmax(dim=1)

#     lm_logits1 = lm_logits1.squeeze()
#     lm_logits2 = lm_logits2.squeeze()
#     # (batch, vocab_size)
#     # give threshold
#     prob2 = softmax(lm_logits2)
#     # the result tuple of two output tensors (max, max_indices)
#     # print(torch.max(prob2, dim=1))
#     prob2_max, prob2_index = torch.max(prob2, dim=1)
#     valid = []
#     for i in range(prob2_max.shape[0]):
#         if (prob2_index[i]==5839 and prob2_max[i]>0.9) or (prob2_index[i]!=5839 and prob2_max[i]>threshold):
#             valid.append(1)
#         else:
#             valid.append(0)

#     #sharpening
#     soft_targets = softmax(lm_logits2/0.5)

#     loss_temp = torch.sum(- soft_targets * logsoftmax(lm_logits1), 1)
#     for i in range(prob2_max.shape[0]):
#         if valid[i]==0:
#             loss_temp[i]=0

#     return torch.mean(loss_temp)



class DST_Seq2Seq(pl.LightningModule):

    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]
        self.flag = 0
        self.save_p = "t5_original"
    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = self.model(input_ids=batch["encoder_input"],
                            # attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]
                            ).loss


      

        #########
        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        # wandb.log({"train_loss": loss})
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss }
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = self.model(input_ids=batch["encoder_input"],
                            # attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]
                            ).loss
        # wandb.log({"val_loss": loss})
        self.log('val_loss', loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    # def validation_epoch_end(self, outputs):
    #     val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
    #     # show val_loss in progress bar but only log val_loss
    #     results = {'progress_bar': {'val_loss': val_loss_mean.item()}}
    #     return results

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)



def eval_from_checkpoint(args, *more):
    args = vars(args)
    # args["model_name"] = args["model_checkpoint"]+args["model_name"]+"_except_domain_"+args["except_domain"]+ "_slotlang_" +str(args["slot_lang"]) + "_joint_" + str(args["joint_training"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    # train!
    args["model_name"] = args["model_checkpoint"] + 't5'
    print(args)
    seed_everything(args["seed"])

    

    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        # model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        # tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        # model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
        tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bart" in args["model_name"]:
        model = BartForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = BartTokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    task = DST_Seq2Seq(args, tokenizer, model)
    task.load_state_dict(torch.load("{}/task.pt".format((args["model_checkpoint"]))))
    _, _, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args, task.tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    save_path = os.path.join(save_path, args["self_training"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
    
    # trainer = Trainer(
    #                 default_root_dir=save_path,
    #                 accumulate_grad_batches=args["gradient_accumulation_steps"],
    #                 gradient_clip_val=args["max_norm"],
    #                 max_epochs=args["n_epochs"],
    #                 callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
    #                 gpus=args["GPU"],
    #                 deterministic=True,
    #                 num_nodes=1,
                    #precision=16,
                    # accelerator="ddp"
                    # )
    # wandb.config = { 
    #     "learning_rate": 1e-4,
    #     "epochs": 5,
    #     "batch_size": 4

        
    # }
    # wandb.watch(task.model)
    # trainer.fit(task, train_loader, val_loader)
    # print(save_path)
    # print(args["joint_training"])
    # task.model.save_pretrained(save_path)
    # task.tokenizer.save_pretrained(save_path)
    # torch.save(task.state_dict(), "{}/task.pt".format(save_path))
    print("test start...")
    #evaluate model
    _, prediction_path = evaluate_eval(args, task.tokenizer, task.model, test_loader, save_path, ALL_SLOTS)
    return prediction_path

def evaluate_eval(args, tokenizer, model, test_loader, save_path, ALL_SLOTS, prefix="zeroshot"):
    save_path = os.path.join(save_path, args["joint_training"])
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}

    for batch in tqdm(test_loader):
        
        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                # turn_att_a=batch["turn_att_a"].to(device),
                                # turn_att_b=batch["turn_att_b"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=200,
                                )

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}

            if value!="none":
                # continue
                if args["self_training"] == 'R2':
                    predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(value)
                else:
                    predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+str(value))
                

            # analyze slot acc:
    #         if str(value)==str(batch["value_text"][idx]):
    #             slot_logger[str(batch["slot_text"][idx])][1]+=1 # hit
    #         slot_logger[str(batch["slot_text"][idx])][0]+=1 # total

    # for slot_log in slot_logger.values():
    #     slot_log[2] = slot_log[1]/slot_log[0]

    # with open(os.path.join(save_path, f"{prefix}_slot_acc.json"), 'w') as f:
    #     json.dump(slot_logger,f, indent=4)

    

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(f"{prefix} result:",evaluation_metrics)
    # wandb.log(evaluate_metrics)
    prediction_path = os.path.join(save_path, "zeroshot_prediction.json")
    with open(os.path.join(save_path, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)
    with open(os.path.join(save_path, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)

    return predictions, prediction_path


def finetune_from_checkpoint(args, *more):
    args = vars(args)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    print('Doing finetune for RATIO of', args['fewshot'])
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    seed_everything(args["seed"])
    domains = ["hotel", "train", "restaurant", "attraction", "taxi"]
    # for domain in domains:
    #     if domain in args["model_checkpoint"]:
    #         args["only_domain"] = domain
    assert args["only_domain"]!="none"
    # args["model_checkpoint"] = os.path.join(args["saving_dir"],args["model_name"])
    print(args)
    args["model_name"] = args["model_checkpoint"] + 't5'
    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    save_path = os.path.join(save_path, args["self_training"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)

    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained('t5-small', bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
    elif "bart" in args["model_name"]:
        model = BartForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = BartTokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")

    task = DST_Seq2Seq(args, tokenizer, model)
    train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args, tokenizer)
    
    
    trainer = Trainer(
                    default_root_dir=args["model_checkpoint"],
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=8,verbose=False, mode='min')],
                    gpus=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    # precision=16,
                    # accelerator="ddp"
                    )

    trainer.fit(task, train_loader, val_loader)
    #save model
    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)
    torch.save(task.state_dict(), "{}/task.pt".format(save_path))

    print("test start...")
    #evaluate model
    ratio = "ratio_" + str(args["fewshot"]) + "_seed_" + str(args["seed"])
    _, prediction_path = evaluate_eval(args, task.tokenizer, task.model, test_loader, save_path, ALL_SLOTS, prefix=ratio)

    return save_path, prediction_path

if __name__ == "__main__":
    args = get_args()
    # wandb.init(project="T5DST", entity="victorli")
    
    if args.mode=="train":
        save_path = eval_from_checkpoint(args)
        print(save_path)

    elif args.mode == 'finetune':
        save_path, prediction_path = finetune_from_checkpoint(args)
        
