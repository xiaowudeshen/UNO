
# from data_loader import prepare_data
from data_loader_final import prepare_data
from config import get_args
from TA_eval import eval_from_checkpoint, finetune_from_checkpoint
from prepare_self_training import data_preprocessing
import json
import time
import os

'''
strong supervision, after using main tasks and sub-tasks to select good labels for finetuining, we further create pseudo labels for sub-tasks

'''

def R1(args, training_data_path, output_file_R1):
    
    start_time = time.time()
    # breakpoint()
    # step 1: From Cross doamin training to predict zero-shot results
    # produce pseudo labels for main task 1
    # if result predicted already, go back to perform value masking
    print('Doing first step of self_training, start at', time.time())
    args.self_training = 'R1'
    ################
    pred_path1 = eval_from_checkpoint(args)
    arg_v = vars(args)
    # pred_path1 = "test_self/t5_flan_joint/google/flan-t5-smallt5_except_domain_"+ arg_v["only_domain"]+ "joint_taskmask_slot_slotlang_question_lr_0.0001_epoch_5_seed_577/mask_slott5/R1/none/results/zeroshot_prediction.json"
    # pred_path1 = "test_self/t5_QA/QA_pret5_except_domain_"+ arg_v["only_domain"]+ "joint_taskmask_slot_slotlang_question_lr_0.0001_epoch_1_seed_577/mask_slott5/R1/none/results/zeroshot_prediction.json"
    ################
    # breakpoint()
    preprocessed_data = data_preprocessing(training_data_path, pred_path1, output_file_R1,  purpose = 'prepare_slot')
    time1 = time.time()
    print(f"first step done, time taken {(time1-start_time)/3600} hrs, result for mask slots are saved here: {output_file_R1}")
    next_step = 'R2'
    return pred_path1



def R2(args, training_data_path, pred_path1, output_file_R2, output_file_R1):


    start_time = time.time()
    time0 = start_time
    # step 2: From masked slot value, use the origianl model to predict the slot type
    # produce pseudo labels for task 2
    # prediction result will be save to inter mediate file
    print('Doing second step of self_training, start at', time.time())
    args.self_training = 'R2'
    # preprocessed_data = data_preprocessing(training_data_path, pred_path1, output_file_R2,  purpose = 'prepare_finetuning')
    # pred_path2 = eval_from_checkpoint(args)
    # pred_path2 = 'test_self/t5_flan_joint/google/flan-t5-smallt5_except_domain_hoteljoint_taskmask_slot_slotlang_question_lr_0.0001_epoch_5_seed_577/mask_slott5/R2/none/results/zeroshot_prediction.json'
    preprocessed_data = data_preprocessing(training_data_path, pred_path1, output_file_R2,  purpose = 'selecting_oracle')
    joint_acc, turn_acc_dict = preprocessed_data.check_slot_prediction_acc(pred_path1, purpose = 'selecting_oracle')
    print(f"joint accuracy is {joint_acc}")
    # print(f"turn accuracy is {turn_acc_dict}")
    # augmented_dialogues = preprocessed_data.dialogue_augmentation(output_file_R1, output_file_R2)
    # breakpoint()
    time2 = time.time()
    time0 = time2
    print(f"second step done, time taken {(time2-start_time)/3600} hrs,  result for mask slots are saved here: {output_file_R2}")
    next_step = 'R3'
    


def R3(args):
    print('Doing third step of self_training, start at', time.time())
    # breakpoint()
    start_time = time.time()
    args.mode = "finetune"
    args.except_domain = 'none'
    # args.fewshot = 1.0
    args.self_training = 'R3'
    save_path1, pred_path3 = finetune_from_checkpoint(args)
    # save_path1, pred_path3 = eval_from_checkpoint(args)
    
    args.model_checkpoint = save_path1
    time3 = time.time()
    print(f"third step done, time taken {(time3-start_time)/3600} hrs, result for mask slots are saved here: {pred_path3}")
    time0 = time3
    # args.fewshot = 0.0
    next_step = 'R1'
    return save_path1, pred_path3




def self_training_step1(args, training_round):
    vargs = vars(args)
    print("Self_training_starts++++++++++++++++++++++++++++++++++++++++++++++")
    training_data_path = "data/train_dials.json"
    #file to save all the predicted results
    output_file_R1 = 'data_self/slot_train_dials_' + vargs["only_domain"] + '.json'
    #file to save all the selected dialogues
    output_file_R2 = 'data_self/Selected_w_dialogs_' + vargs["only_domain"] + '.json'

    if not os.path.exists('data_self'):
        os.makedirs('data_self')
    total_rounds = training_round
    next_step = vargs["next_step"]
        
    time0 = time.time()
    for i in range(total_rounds):
            
        # step 1: From Cross doamin training to predict zero-shot results
        # produce pseudo labels for main task 1
        # if result predicted already, go back to perform value masking
        if next_step == 'R1':
            pred_path1 = R1(args, training_data_path, output_file_R1)
            next_step = 'R2'
        # breakpoint()
        # pred_path1 = "t5-self/t5_flan/google/flan-t5-smallt5_except_domain_hotel_slotlang_question_lr_0.0001_epoch_5_seed_577t5/R1/none/results/zeroshot_prediction.json"
        # step 2: From masked slot value, use the origianl model to predict the slot type
        # produce pseudo labels for task 2
        # prediction result will be save to inter mediate file
        if next_step == 'R2':
            preprocessed_data = R2(args, training_data_path, pred_path1, output_file_R2, output_file_R1)
            next_step = 'R3'

        # breakpoint()

        
        # step 3: From the good labels selected with consistent result of task 1 and task 2
        # perform a finetuneing of the original dataset
        if next_step == 'R3':
            save_path1, pred_path3 = R3(args)
            next_step = 'R2'  
            pred_path1 = pred_path3
            args.model_checkpoint = save_path1

    print('Self-training done, total time taken is: ', (time.time()-time0)/3600, 'hrs')





        
        


if __name__ == "__main__":
    args = get_args()
    total_rounds = 1
    if args.mode == "self_training":
        self_training_step1(args, total_rounds)