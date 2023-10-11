from builtins import breakpoint
import pandas as pd
import json
import copy
import random



class data_preprocessing():
    def __init__(self, data_json_file, out_file_path):
        self.data = self.read_json(data_json_file)
        self.mask = "<extra_id_0>"
        self.dup_list = []
        self.dup_dic = {}
        self.updated_data = []
        for dialogue in self.data:
            self.new_dialogue = dialogue.copy()
            self.replace_values(dialogue)
            self.updated_data.append(self.new_dialogue)


        self.save(out_file_path)

    def replace_values(self, dialogue ):
        '''
        replaces a given slot value with a different slot value
        if you dont want this function to edit the original data, pass a deepcopy of your dialogue
        '''
        dialog_history = ""
        
        
        for idx, turn in enumerate(dialogue['turns']):
            
            
            if self.check_duplicates(turn):
                self.dup_list.append(dialogue['dial_id'])
                self.dup_dic[dialogue['dial_id']] = 1
                dup_keys = self.return_dup_keys(turn)
                dup_list = []
                for item in dup_keys:
                    a = list(item)
                    dup_list += a

            mask_list = []
            
            for slot in turn['state']['turn_slot_values'].keys():
                # if dialogue['dial_id'] == "MUL2168.json":
                #     breakpoint()
                #     print(slot)
                if self.check_duplicates(turn) and slot in dup_list:
                    print('=============')
                    print('Duplicated values found in dialogue', dialogue["dial_id"])
                    
                    continue
                 
                temp_dict = {}
                temp_dict['user'] = turn['user']
                temp_dict['system'] = turn['system']
                if turn['state']['turn_slot_values'][slot] in turn['user']:
                    temp_dict['user'] = turn['user'].replace(turn['state']['turn_slot_values'][slot], self.mask)
                elif turn['state']['turn_slot_values'][slot] in turn['system']:
                    temp_dict['system'] = turn['system'].replace(turn['state']['turn_slot_values'][slot], self.mask)
                temp_history = dialog_history + (" System: " + temp_dict["system"] + " User: " + temp_dict["user"]) 
                temp_dict["dialog_history"] = temp_history
                temp_dict["masked_type"] = slot
                mask_list.append(temp_dict)

            
            dialog_history += (" System: " + turn["system"] + " User: " + turn["user"])
                
            self.new_dialogue["turns"][idx]["masked_state"] = mask_list
             
            

                
    

    def check_duplicates(self, turn):
        values = turn['state']['turn_slot_values'].values()
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

    def save(self, path):
        '''
        saves the dictionary as a json at the desired location
        '''
        out_file = open(path, 'w+')
        
        json.dump(self.updated_data, out_file, indent=4)
        print(len(self.dup_dic))
        print(self.dup_dic)

        


preprocessed_data = data_preprocessing("data1/new_train_dials.json",  "data1/mask_train_dials.json")
preprocessed_data = data_preprocessing("data1/new_dev_dials.json",  "data1/mask_dev_dials.json")
preprocessed_data = data_preprocessing("data1/new_test_dials.json",  "data1/mask_test_dials.json")