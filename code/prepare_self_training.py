from builtins import breakpoint
import pandas as pd
import json
import copy
import random
from evaluate import evaluate_metrics


class data_preprocessing():
    #input1: original datafile with the testing data
    #input2: model_checkpoint, the directory where the result is stored
    #output: a new file where the processed masked slot type is stored
    def __init__(self, data_json_file, prediction_file, out_file_path,  statistical_path = 'data_self/Selection_data.json', purpose = 'prepare_slot'):
        self.purpose = purpose
        self.data = self.read_json(data_json_file)
        self.data1 = self.read_json(prediction_file)
        # if slot_pred is not None:
        #     self.spred = self.read_json(slot_pred)
        DOMAINS = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
        for domain in DOMAINS:
            if domain in prediction_file:
                self.domain = domain
                break
        self.mask = "<extra_id_0>"
        self.dial_dic = {}
        
        self.dup_list = []
        self.dup_dic = {}
        self.updated_data = []
        if self.purpose == 'prepare_slot':
            print(f'Selecting dialogue that can be used to generate mask slot type, dialogue save in {out_file_path}, statistical data saved in {statistical_path}')
            self.selection_matrix = {}
            self.selection_matrix = {
            "pred_not_in_turn":0,
            "pred_in_turn":0,
            "total_pred":0,
            "pred_duplicated":0,
            "Evaluation_of_sampes":{}
            }
            predictions = {}
            #convert original test_data list into a dictionary
            for dialog in self.data:
                self.dial_dic[dialog["dial_id"]] = dialog
            # for each prediction, prepare for the mask slot type
            for pred_id, prediction in self.data1.items():
                # breakpoint()
                dialogue = self.dial_dic[pred_id]
                new_dialogue = dialogue.copy()
                # print(dialogue["dial_id"])
                self.new_dialogue = self.replace_values(new_dialogue, prediction)
                self.updated_data.append(self.new_dialogue)

                ### add evaluation metrics to all predictions
                # ALL_SLOTS = {}
                # predictions[pred_id] = dialogue
            # joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(self.data1, ALL_SLOTS)

            # evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
            # self.selection_matrix["Evaluation_of_sampes"] = evaluation_metrics

            # breakpoint()
            self.save(self.updated_data, out_file_path)
            self.save(self.selection_matrix, statistical_path)

        elif self.purpose == 'prepare_finetuning':
            print('Selecting dialogue for self_training')
            if prediction_file is None:
                raise "You have to link the slot type prediction file"
            new_slot_dict = self.check_for_new_slot()
            if new_slot_dict != {}:
                print("There are new slots in the prediction file, please check the new slot dictionary")
            total_accuracy, none_accuracy, label_list, label_dic = self.generate_good_labels()
            evaluation_metrics = {"Total Accuracy for slot matching":total_accuracy, "none_ratio":none_accuracy, "label_list":label_list, "label_dic": label_dic, "new_slot_dict":new_slot_dict}
            print("Length of good labels:", len(label_list))
            print("Length of training data", len(self.data))
            print("Length of in domain training data", len(self.data1))
            print("Ratio of good labels", len(label_list)/len(self.data))
            # print("Evaluation metrics:", evaluation_metrics)
            self.save(evaluation_metrics, out_file_path)

        elif self.purpose == 'select_not_none':
            print('Select values that are not none for self-training')
            if prediction_file is None:
                raise "You have to link the prediction file for selection"
            total_accuracy, none_accuracy, label_list, label_dic = self.select_not_none_labels()
            evaluation_metrics = {"Average slot per turn":total_accuracy, "none_ratio":none_accuracy, "label_list":label_list, "label_dic": label_dic}
            print("Length of good labels:", len(label_list))
            print("Length of training data", len(self.data))
            print("Length of in domain training data", len(self.data1))
            print("Ratio of good labels", len(label_list)/len(self.data))
            self.save(evaluation_metrics, out_file_path)

        elif self.purpose == 'selecting_oracle':
            print('Selecting dialogue for oracle self_training results')
            if prediction_file is None:
                raise "You have to link the slot type prediction file"
            new_slot_dict = self.check_for_new_slot()
            if new_slot_dict != {}:
                print("There are new slots in the prediction file, please check the new slot dictionary")
            total_accuracy, none_accuracy, label_list, label_dic = self.generate_good_labels()
            evaluation_metrics = {"Total Accuracy for slot matching":total_accuracy, "none_ratio":none_accuracy, "label_list":label_list, "label_dic": label_dic, "new_slot_dict":new_slot_dict}
            print("Length of good labels:", len(label_list))
            print("Length of training data", len(self.data))
            print("Length of in domain training data", len(self.data1))
            print("Ratio of good labels", len(label_list)/len(self.data))
            # print("Evaluation metrics:", evaluation_metrics)
            self.save(evaluation_metrics, out_file_path)
        else: 
            print('You need to select a purpose for this script, with prepare_slot or prepare_finetuning')

    def generate_oracle_results(self):
        '''
        compared with the ground_truth in the training data, genenrate the oracle results for good predictions
        '''
        label_list = []
        label_dic = {}
        # joint_acc = 0
        total_count = 0
        total_acc = 0
        total_non = 0
        for idx, dial in self.data1.items():
            turn_acc = 0
            turn_count = 0
            for k, cv in dial["turns"].items():
                if set(cv["pred_belief"]).issubset(set(cv["turn_belief"])):
                    # breakpoint()
                    if cv["turn_belief"] == []:
                        total_non += 1
                    # joint_acc += 1
                    turn_acc += 1
                    total_acc += 1
                turn_count += 1
                total_count += 1
                # total_count += 1
            turn_accuracy = turn_acc/float(turn_count) if turn_count !=0 else 0

            if turn_accuracy == 1:
                label_list.append(idx)
            
            label_dic[idx] = turn_accuracy
        none_accuracy = total_non/float(total_count) if total_count !=0 else 0
        total_accuracy = total_acc/float(total_count) if total_count !=0 else 0
        return total_accuracy, none_accuracy, label_list, label_dic


    def generate_good_labels(self):
        label_list = []
        label_dic = {}
        # joint_acc = 0
        total_count = 0
        total_acc = 0
        total_non = 0
        for idx, dial in self.data1.items():
            turn_acc = 0
            
            turn_count = 0
            for k, cv in dial["turns"].items():
                if set(cv["turn_belief"]) == set(cv["pred_belief"]):
                    # breakpoint()
                    if cv["turn_belief"] == []:
                        total_non += 1
                    # joint_acc += 1
                    turn_acc += 1
                    total_acc += 1
                turn_count += 1
                total_count += 1
                # total_count += 1
            turn_accuracy = turn_acc/float(turn_count) if turn_count !=0 else 0

            if turn_accuracy == 1:
                label_list.append(idx)
            
            label_dic[idx] = turn_accuracy
        none_accuracy = total_non/float(total_count) if total_count !=0 else 0
        total_accuracy = total_acc/float(total_count) if total_count !=0 else 0
        return total_accuracy, none_accuracy, label_list, label_dic
            
    


    def select_not_none_labels(self):
        label_list = []
        label_dic = {}

        total_count = 0
        total_acc = 0
        total_non = 0
        for idx, dial in self.data1.items():
            turn_acc = 0
            
            turn_count = 0
            for k, cv in dial["turns"].items():
                turn_acc += len(cv["pred_belief"])
                total_acc += len(cv["pred_belief"])
                turn_count += 1
                total_count += 1 
                if len(cv["pred_belief"])==0:
                    total_non += 1
            turn_accuracy = turn_acc/float(turn_count) if turn_count !=0 else 0

            if turn_accuracy >= 1:
                label_list.append(idx)
            
            label_dic[idx] = turn_accuracy
        none_accuracy = total_non/float(total_count) if total_count !=0 else 0
        total_accuracy = total_acc/float(total_count) if total_count !=0 else 0
        return total_accuracy, none_accuracy, label_list, label_dic

        
    def check_slot_prediction_acc(self, R1_pred_path,  purpose = 'prepare_finetuning' ):
        '''
        check the JGA of the selected dialogue
        '''
        if purpose == 'prepare_finetuning':
            _, _, labels_list, _ = self.generate_good_labels()
        elif purpose == 'select_not_none':
            _, _, labels_list, _ = self.select_not_none_labels()
        elif purpose == 'selecting_oracle':
            _, _, labels_list, _ = self.generate_oracle_results()
        total_count = 0
        total_acc = 0
        turn_acc_dict = {}
        R1_data = self.read_json(R1_pred_path)
        for idx in labels_list:
            dial = R1_data[idx]
            turn_acc = 0
            turn_count = 0  
            for k, cv in dial["turns"].items():
                if set(cv["turn_belief"]) == set(cv["pred_belief"]):
                    turn_acc += 1
                    total_acc += 1
                turn_count += 1
                total_count += 1
            turn_accuracy = turn_acc/float(turn_count) if turn_count !=0 else 0
            turn_acc_dict[idx] = turn_accuracy
        joint_acc = total_acc/float(total_count) if total_count !=0 else 0
        return joint_acc, turn_acc_dict







    def replace_values(self, dialogue, prediction):
        '''
        replaces a given slot value with a different slot value
        if you dont want this function to edit the original data, pass a deepcopy of your dialogue
        '''
        dialog_history = ""
        
        new_dialogue = copy.deepcopy(dialogue)
        for idx, turn in enumerate(dialogue['turns']):
            # breakpoint()
            # if len(prediction["turns"])<= idx:
            #     continue
            preds = prediction["turns"][str(idx)]["pred_belief"]
            # print(preds )
            
            
            new_dialogue["turns"][idx]["state"]["slot_values"] = {f"{k.split('-')[0]}-{k.split('-')[1]}" :  k.split('-')[-1] for k in preds}
            mask_list = []
            if self.check_duplicates(preds):
                self.dup_list.append(dialogue['dial_id'])
                self.dup_dic[dialogue['dial_id']] = 1
                self.selection_matrix["pred_duplicated"] +=1
                
                
            else:    

                
                
                for pred in preds:
                    temp_dict = {}
                    temp_dict['user'] = turn['user']
                    temp_dict['system'] = turn['system']
                    dialog_history += (" System: " + turn["system"] + " User: " + turn["user"])
                    if pred.split('-')[-1] in dialog_history:
                        temp_history = dialog_history.replace(pred.split('-')[-1], self.mask)
                        temp_dict["masked_type"] = pred.split('-')[0] + '-' +  pred.split('-')[1]
                        temp_dict["dialog_history"] = temp_history
                        mask_list.append(temp_dict)
                        self.selection_matrix["pred_in_turn"] += 1
                        self.selection_matrix["total_pred"]+= 1
                 
                    else:
                        self.selection_matrix["pred_not_in_turn"] += 1
                        self.selection_matrix["total_pred"]+= 1
            new_dialogue["turns"][idx]["masked_state"] = mask_list    

        return new_dialogue




    def retrieve_values(self, dialogue, prediction):
        '''
        Almost the same with previous replace value function, 
        The only difference is that we prepare value_slot and origianl dialogue history for the value_slot joint task

        new promopt, what is the slot type with the value XX?
        '''
        dialog_history = ""
        
        new_dialogue = copy.deepcopy(dialogue)
        for idx, turn in enumerate(dialogue['turns']):
            # breakpoint()
            # if len(prediction["turns"])<= idx:
            #     continue
            preds = prediction["turns"][str(idx)]["pred_belief"]
            # print(preds )
            
            
            new_dialogue["turns"][idx]["state"]["slot_values"] = {f"{k.split('-')[0]}-{k.split('-')[1]}" :  k.split('-')[-1] for k in preds}
            mask_list = []
            if self.check_duplicates(preds):
                self.dup_list.append(dialogue['dial_id'])
                self.dup_dic[dialogue['dial_id']] = 1
                self.selection_matrix["pred_duplicated"] +=1
                #
                # create another copy of the dialogue with the duplicates
                # eg : dialog_history + slot_p + value ?
                # slot type: slot_type_A + slot_type_B + slot_type_C
            else:    
                
                for pred in preds:

                    


                    temp_dict = {}
                    temp_dict['user'] = turn['user']
                    temp_dict['system'] = turn['system']
                    dialog_history += (" System: " + turn["system"] + " User: " + turn["user"])
                    if pred.split('-')[-1] in dialog_history:
                        #keep the original dialogue history
                        temp_history = dialog_history
                        temp_dict["slot_value"] = pred.split('-')[-1]
                        temp_dict["masked_type"] = pred.split('-')[0] + '-' +  pred.split('-')[1]
                        temp_dict["dialog_history"] = temp_history
                        mask_list.append(temp_dict)
                        self.selection_matrix["pred_in_turn"] += 1
                        self.selection_matrix["total_pred"]+= 1
                 
                    else:
                        self.selection_matrix["pred_not_in_turn"] += 1
                        self.selection_matrix["total_pred"]+= 1
            new_dialogue["turns"][idx]["masked_state"] = mask_list    

        return new_dialogue


    def dialogue_augmentation(self,  prediction_dialogues, label_file):
        '''
        Augments the selected dialogues with different values for each slot
        '''
        # breakpoint()
        print("Augmenting dialogues")
        data = self.read_json(label_file)
        selection_list = data["label_list"]
        pred_data = self.read_json(prediction_dialogues)
        selected_dialogues = [k for k in pred_data if k["dial_id"] in selection_list]

        augmented_dialogues = []
        slot_values = {}
        for dialogue in selected_dialogues:
            for turn in dialogue['turns']:
                # breakpoint()
                for slot, value in turn['state']['slot_values'].items():
                    if slot in slot_values:
                        if value not in slot_values[slot]:
                            slot_values[slot].append(value)
                    else:
                        slot_values[slot] = [value]
                    

        '''
        we go back to randomly replace the original value of each slot in dialogues with different values, and output them
        in the same format as the origianl json file
        '''
        for dialogue in selected_dialogues:
            aug_dialogue = copy.deepcopy(dialogue)
            aug_dialogue['turns'] = []
            for idx, turn in enumerate(dialogue['turns']):
                augmented_turn = copy.deepcopy(turn)
                augmented_turn['state']['slot_values']= {}
                augmented_turn['masked_state'] = []
                for slot, value in turn['state']['slot_values'].items():
                    if len(slot_values[slot]) > 1:
                        new_value = random.choice(slot_values[slot])
                        while new_value == value:
                            new_value = random.choice(slot_values[slot])
                        augmented_turn['state']['slot_values'][slot] = new_value
                        for masked in turn['masked_state']:
                            if masked['masked_type'] == slot:
                                aug_masked = copy.deepcopy(masked)
                                aug_masked['slot_value'] = new_value
                                aug_masked['dialog_history'] = aug_masked['dialog_history'].replace(self.mask, new_value)
                                # augmented_turn['masked_state'].append(masked)
                                augmented_turn['masked_state'].append(aug_masked)
                            
                aug_dialogue['turns'].append(augmented_turn)

            augmented_dialogues.append(aug_dialogue)
        augmented_dialogues.append({'slot_values': slot_values})
        self.save(augmented_dialogues, f"data_self/augmented_dialogues_{self.domain}.json")       
        print("Augmented dialogues are saved in data_self/augmented_dialogues_{self.domain}.json")


        return augmented_dialogues
            


    def check_for_new_slot(self):
        '''
        checks if a new slot(other than the original domain-slot in the target domain) is added to the dialogue
        '''
        new_slot_dict = {}
        slot_description = self.read_json('utils/slot_description.json')
        old_slot_list = []
        for k in slot_description.keys():
            if self.domain in k:
                old_slot_list.append(k)
        # old_slot_list = list(k for k in slot_description.keys())
        for idx, dial in self.data1.items():
            for k, cv in dial["turns"].items():
                for pred in cv["pred_belief"]:
                    if pred not in old_slot_list:
                        if pred in new_slot_dict:
                            new_slot_dict[pred].append(f"dial:{idx} -turn:{k}")
                        else:
                            new_slot_dict[pred] = [f"dial:{idx} -turn:{k}"]
        return new_slot_dict
        

                
    

    def check_duplicates(self, pred_values):
        values = [pred.split('-')[-1] for pred in pred_values]
        if len(values) == len(set(values)):
            return False
        else:
            return True

    def return_dup_keys(self, turn):
        rev_multidict = {}
        for key, value in turn['state']['turn_slot_values'].items():
            rev_multidict.setdefault(value, set()).add(key)
        dup_keys = [values for key, values in rev_multidict.items() if len(values) > 1]
        

        return dup_keys
        

    def read_json(self, name):
        '''
        input: a json file path
        reads a json file as a dictionary
        '''
        f = open(name)
        data = json.load(f)

        return data

    def save(self, content, path):
        '''
        saves the dictionary as a json at the desired location
        '''
        out_file = open(path, 'w+')
        
        json.dump(content, out_file, indent=4)
        
        # print(len(self.dup_dic))
        # print(self.dup_dic)









if __name__ == "__main__":
    original_test_data_file = "data/test_dials.json"
    training_data_path = 'data/train_dials.json'
    pred_path1 = 'QA_result_qtype/QA_pret5_except_domain_hotel_slotlang_question_joint_slot_epoch_1_seed_11/slot/results/zeroshot_prediction.json'
    output_file_R2 = 'data1/Selected_w_dialogs_none'
    preprocessed_data = data_preprocessing(training_data_path, pred_path1, output_file_R2,  purpose = 'select_not_none')
# preprocessed_data = data_preprocessing("data/train_dials.json", "self_result_train/pp_result/pptod-smallt5_except_domain_hotel_slotlang_slottype_joint_slot_1_epoch_1_seed_11t5_except_domain_hotel_slotlang_slottype_joint_slot_1_epoch_20_seed_11/slot_1/results/zeroshot_prediction.json" ,"data1/slot_train_dials.json")
# preprocessed_data = data_preprocessing("data/train_dials.json", slot_pred_file, output_file,  purpose = 'prepare_finetuning')