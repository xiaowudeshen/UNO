# UNO-DST: Leveraging Unlabelled Data in Zero-shot Dialogue State Tracking

## Abstract:
Previous zero-shot dialogue state tracking (DST) methods only apply transfer learning while ignoring unlabelled data in the target domain.
To mitigate this, we transform zero-shot DST into few-shot DST via joint and self-training methods. Our method incorporates auxiliary tasks that generate slot types as inverse prompts for main tasks, creating slot values during joint training. The cycle consistency between these two tasks enables the generation and selection of quality samples in unknown target domains for subsequent fine-tuning. This approach also facilitates automatic label creation,
thereby optimizing the training and fine-tuning of DST models. We demonstrate the effectiveness and potential of this method on large language models. Experimental outcomes demonstrate our method's efficacy in zero-shot scenarios by improving average joint goal accuracy by 8% across all domains in MultiWOZ. 

## Method:
<p align="center">
<img src="figures/diagram.png" width="100%" />

</p>


## Citation


## Baseline
Check our baseline on T5DST from Facebook research: [GIT REPO](https://github.com/facebookresearch/Zero-Shot-DST/tree/main/T5DST). Our code is modified based on the T5DST official repo.

## Environment
Install the environment from the provided "env" file
```console
❱❱❱ conda env create -f UNO-DST_env.yml
```

## Experiments
**Dataset**
```console
❱❱❱ python create_data.py
```
use create_data_2_1.py if you want to run with multiwoz2.1

**Data Preprocessing**
```console
❱❱❱ python preprocessing_new.py
❱❱❱ python prepare_mask_pretrain.py
```
preprocessing_new.py check if data is in correct format and check if the turn slot value are correct
prepare_mask_pretrain.py perform masking of the training data, as in joint training period


**Joint Traning Period**
```console
❱❱❱ # python T5.py --train_batch_size 8 --GPU 1 --n_epochs 1 --model_checkpoint t5-small --saving_dir t5_small --slot_lang question --except_domain ${domain} --joint_training mask_slot 
```
* --GPU: the number of gpu to use
* --except_domain: hold out domain, choose one from [hotel, train, attraction, restaurant, taxi]

**Self-Traning Period**
```console
❱❱❱ python self_step2.py --train_batch_size 8 --GPU 1 --mode "self_training" --slot_lang question --saving_dir t5_self --n_epochs 3 --only_domain  ${domain} --next_step "R1" --model_checkpoint ${model_checkpoint}
```
* --model_checkpoint: directory for saved model weights and config file
* --next_step: next step training step, choose from R1, R2, R3
  
**Oracle Results for any Baseline or Checkpoint**
```console
❱❱❱ python self_step_oracle.py --train_batch_size 8 --GPU 1 --mode "self_training" --slot_lang question --saving_dir t5_self_oracle --n_epochs 1 --only_domain $domain --next_step "R1" --model_checkpoint ${model_checkpoint}
```
* --model_checkpoint: directory for saved model weights and config file
