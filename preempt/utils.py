import json
import random
from typing import Dict, List, Any, Optional, Union
import re
import unicodedata
import ast
from names_dataset import NameDataset

from .common import NAME_PREFIXES
from .conversation import get_conv_template, register_conv_template, Conversation, SeparatorStyle
import numpy as np
import torch
from torch.utils.data import Dataset


"""
###################################################
#################   Pipelining    #################
###################################################
"""
"""
Register template for UniNER models.
"""
register_conv_template(
    Conversation(
        name="ie_as_qa",
        system_message="A virtual assistant answers questions from a user based on the provided text.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

def check(name_list: List[str], target: str):
    """
    For checking if a target name exists in a given list of names
    """
    for i, n in enumerate(name_list):
        if finder(name_list[i], target):
            return i
def finder(word1, word2):
    encoded1 = unicodedata.normalize('NFC', word1)
    encoded2 = unicodedata.normalize('NFC', word2)
    return encoded1==encoded2

def make_names_dataset(save_path:str) -> None:
    """
    For making a new dataset of names for sanitizing names.
    """
    nd = NameDataset()

    countries = ['US','GB','FR']
    first_names = []
    last_names = []

    for country in countries: 
        first_names = first_names + nd.get_top_names(n=1000,gender='M',country_alpha2=country)[country]['M']
    for country in countries: 
        last_names = last_names +   nd.get_top_names(n=1000,country_alpha2=country,use_first_names=False)[country]

    # Get rid of duplicates
    remove_fns = ['Saba','Lanka','Deblog', 'Donas', 'Bk']
    remove_lns = ['guez','quez','ecour','Mai','ü', 'é', 'Behal', 'Bk']
    first_names = list(set(first_names))
    last_names = list(set(last_names))

    temp = []
    for i in range(len(first_names)):
        flag = 0
        for rfn in remove_fns:
            if rfn in first_names[i]: 
                flag = 1
                break
        if flag==0:
            temp.append(first_names[i])

    first_names = temp[:]
    for i in range(len(last_names)):
        flag = 0
        for rfn in remove_lns:
            if rfn in last_names[i]: 
                flag = 1
                break
        if flag==0:
            temp.append(last_names[i])

    last_names = temp[:]
    save_fn({"first_names": first_names, "last_names": last_names}, save_path)

def preprocess_instance(source: List[str]):
    """
    This method is referenced from https://github.com/universal-ner/universal-ner
    Required for using the UniNER model
    """
    conv = get_conv_template("ie_as_qa")
    for j, sentence in enumerate(source):
        value = sentence['value']
        if j == len(source) - 1:
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt

def get_response(responses: List[str]):
    """
    This method is referenced from https://github.com/universal-ner/universal-ner
    Required for using the UniNER model
    """
    responses = [r.split('ASSISTANT:')[-1].strip() for r in responses]
    return responses

def llama_prompt_preprocessor(all_text: List[str], **kwargs) -> List[str]:
    """
    Prompt preprocessing for NER with Llama-3 8B.
    """
    entity = kwargs['entity_type']
    prompts = []
    for input in all_text:
        conv = get_conv_template('llama-3')
        if entity=='Age':
            conv.set_system_message(
                f"Please identify words in the sentence that can be categorized as '{entity}'. Format the output as list with no additional text such as 'years' or 'aged'. If no words are found, return an empty list. Example: []" 
            )
        elif entity=='Name':
            conv.set_system_message(
                f"Please identify words in the sentence that can be categorized as '{entity}'. Format the output as list with no additional text. Example: ['{entity} 1', '{entity} 2']. If no words are found, return an empty list. Example: []"
            )
        elif entity=='Money':
            conv.set_system_message(
                f"Please identify Currency Value from the given text. DO NOT ADD '000' TO ANY VALUE. DO NOT REMOVE SPACES IN ANY VALUE. Format the output as list with no additional text. Example: ['Currency Value 1', 'Currency Value 2']. If no words are found, return an empty list. Example: []"
            )
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        prompts.append(text)
    return prompts

def uniner_prompt_preprocessor(all_text: List[str], **kwargs) -> List[str]:
    """
    Prompt preprocessing for NER with UniNER.
    """
    entity_type = kwargs['entity_type']
    examples = [
        {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"}, 
                {"from": "gpt", "value": "I've read this text."}, 
                {"from": "human", "value": f"What describes {entity_type} in the text?"}, 
                {"from": "gpt", "value": "[]"}
            ]
        } for text in all_text
    ]
    return [preprocess_instance(example['conversations']) for example in examples]

def gen_delimiters(model_path: str) -> str:
    """
    Some standard delimiters used with the generate() method
    for huggingface models. Add more as needed.
    """
    delimiters = {
        "llama": "assistant\n\n",
        "gemma": "\nmodel\n",
        "uniner": "ASSISTANT:",
        "universal": "ASSISTANT:",

    }
    for key in delimiters:
        if key in model_path.lower():return delimiters[key]

def prompt_preprocessor(model_path: str):
    """
    Prompt preprocesssing directory for NER.
    """
    if 'uniner' in model_path.lower() or 'universal' in model_path.lower():
        return uniner_prompt_preprocessor
    if 'llama' in model_path.lower():
        return llama_prompt_preprocessor
    return None

def clean_name_prefixes(all_text):
    """
    Helper for postprocessing names of people.
    """
    outputs = all_text
    prefixes = NAME_PREFIXES
    for prefix in prefixes:
        outputs = [output.replace(prefix, '') for output in outputs]
    outputs = ['"[' + output.replace("'","") + ']"' for output in outputs]
    return outputs

def postprocess_output(outputs: List[str], output_dict: Dict[str, List[List[str]]], entity_type: str):
    """
    Output postprocessing for entities detected by NER.
    """
    if entity_type=="Name" or entity_type=="Full Name":
        outputs = clean_name_prefixes(outputs) 
    if entity_type=="Age":
        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
    if entity_type=="Zipcode":
        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
    if entity_type=="Money":
        outputs = re.findall(r"[-+]?(?:\d*\.*\,*\s*\d+)", outputs[0])
        outputs = [output.strip() for output in outputs]
    for output in outputs:
        if len(output) > 0: 
            if entity_type=="Money":
                output_dict[entity_type].append(outputs)
                return output_dict
            elif entity_type=="Name" or entity_type=="Full Name":
                temp = unicodedata.normalize("NFKD", ast.literal_eval(output))
                temp = temp.replace('[', '')
                temp = temp.replace(']', '')
                temp = temp.split(", ")
                output_dict[entity_type].append(temp)
            else:
                temp = output.replace('[', '')
                temp = temp.replace(']', '')
                output_dict[entity_type].append(ast.literal_eval(output))

        if output=='[]':
            outputs = [None]
            output_dict[entity_type].pop()
            output_dict[entity_type].append(outputs)

    if len(outputs)==0:
        outputs.append(None)
        output_dict[entity_type].append(outputs)

    return output_dict

class any2en(Dataset):
    """
    Returns an object in the correct format for initializing a torch DataLoader object.
    """
    def __init__(self, text: List[Any]) -> Any:
        self.text = text
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return self.text[idx]

"""
###################################################
################# Pretty printing #################
###################################################
"""
def print_block(vals: str) -> None:
    for val in vals:
        pprint(val)
    print('#'*30)

def pprint(tag: str, val: str) -> None:
    """
    Pretty print a list of values and corresponding text.
    """
    if isinstance(val, str):
        print(f"{tag} {'.'*(30 - len(val) -len(tag))} {val}")
    else:
        print(f"{tag} {'.'*(30 - 5 -len(tag))} {val:.3f}")

"""
####################################################################
################# Data loading and preprocessing   #################
####################################################################
"""
def seed_everything(seed: int) -> None:
    """
    Seeds torch and numpy for deterministic outputs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_fn(dataset: Dict[str, Any], fp: str) -> None:
    """
    Save a JSON-formatted dictionary to a file path.
    """
    with open(fp, 'w') as fp:
        json.dump(dataset, fp, indent=2, sort_keys=True)

def load_data(fp: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    """
    with open(fp, 'r') as fp:
          data = json.load(fp)
    return data  
