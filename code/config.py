# Copyright (c) Facebook, Inc. and its affiliates
# Code modified from the origianl T5DST work
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--meta_batch_size", type=int, default=1, help="Batch size for meta training")
    parser.add_argument("--dev_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=577, help="Random seed")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=2, help="max number of turns in the dialogue")
    parser.add_argument("--GPU", type=int, default=8, help="number of gpu to use")
    parser.add_argument("--model_name", type=str, default="t5", help="use t5 or bart?")
    parser.add_argument("--slot_lang", type=str, default="none", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' or 'binary' slot description")
    parser.add_argument("--fewshot", type=float, default=0.0, help="data ratio for few shot experiment")
    parser.add_argument("--fix_label", action='store_true')
    parser.add_argument("--except_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--semi", action='store_true')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--turn_att", type=str, default="euclidean")
    parser.add_argument("--max_dist", type=int, default=100)
    parser.add_argument("--joint_training", type=str, default="none", help="slot_type_discriminator_summary")
    parser.add_argument("--next_step", type=str, default="R1", help="next round of self-training, user R1 or R2 or R3")
    parser.add_argument("--self_training", type=str, default="R1", help="current round of self-training, user R1 or R2 or R3")


    args = parser.parse_args()
    return args
