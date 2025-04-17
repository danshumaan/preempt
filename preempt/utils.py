import json
import math
import os
import random
import sys
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import re
import unicodedata
import ast
from pyfpe_ff3 import FF3Cipher
from names_dataset import NameDataset
import names

from .conversation import get_conv_template, register_conv_template, Conversation, SeparatorStyle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


"""
###################################################
################# Global variables ################
###################################################
"""
NAME_PREFIXES = ['Prof ', 'Prof.', 'Prof', 'Mrs. ', 'Mrs ', 'Mrs.', 'Mrs', 'Mr ', 'Mr. ', 'Mr.', 'Mr', 'Ms. ', 'Ms.', 'Ms', 'Mrs ', 'Mrs', 'Ms ', 'Ms', 'Herr ', 'Herrn ', 'Frau ', 'M. ', 'Mme. ', 'M ', 'Mme ', 'Madame ', 'Monsieur ', 'Monsieur ', '\"', '"', '[', ']', '\\ ', 'Dr. ', 'Dr.', 'Miss ', 'Miss']
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

"""
###################################################
#################   Pipelining    #################
###################################################
"""
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

class NER():
    """
    Returns a NER model object. Call to run NER for a specific entity type.
    Pass in the delimiter for your model's generation. For example, UniNER
    uses 'ASSISTANT' as the generation delimiter,
    """
    def __init__(self, model_path, delimiter="ASSISTANT:", device='cuda:0'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.delimiter = gen_delimiters(model_path)
        self.preprocessing = prompt_preprocessor(model_path)

    def collator(self, batch: List[str]):
        """
        For data loading.
        """
        tokenized_text = self.tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False
                                  )
        return tokenized_text

    def extract(self, prompts: List[str], entity_type: str, batch_size=1, output_dict: Optional[Dict[str, List[List[str]]]]=None) -> Dict[str, List[List[str]]]:
        """
        Takes in a list of strings and returns a dictionary of PII categories and list of values. 

        Args:
            prompts (List[str]): List of input prompts.
            entity_type (str): Entity for NER extraction. Choose from {Name/Money/Age}
            batch_size (int): Batch size for processing NER.
            output_dict (Optional[Dict[str, List[str]]]): A dictionary with the sensitive attribute as the key 
                                                          with a corresponding list of sensitive values detected
                                                          by NER for every input string. 
        Returns:
            output_dict Dict[str, List[str]]: A dictionary with the sensitive attribute as the key 
                                              with a corresponding list of sensitive values detected
                                              by NER for every input string.
        """
        if output_dict is None: output_dict = dict()
        output_dict[entity_type] = []
        if self.preprocessing: prompts = self.preprocessing(prompts, entity_type=entity_type)
        prompts = any2en(prompts)
        prompt_dataloader = DataLoader(prompts, collate_fn=self.collator, batch_size=batch_size)

        for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            with torch.no_grad():
                prompt_batch.to(self.device)
                outputs = self.model.generate(**prompt_batch, 
                                        do_sample=False,
                                        max_length=4096)
                outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [output.replace("['", "") for output in outputs]
                outputs = [output.replace("']", "") for output in outputs]
                outputs = [r.split(self.delimiter)[-1].strip() for r in outputs]
                output_dict = postprocess_output(outputs, output_dict, entity_type)

        return output_dict
    
class Sanitizer():
    """
    Returns a sanitizer object. Call to run sanitization for a list of strings.
    """
    def __init__(self, ner_model: NER, key = "EF4359D8D580AA4F7F036D6F04FC6A94", tweak = "D8E7920AFA330A73"):
        self.cipher_fn = FF3Cipher(key, tweak, allow_small_domain=True, radix=10)
        self.nd = NameDataset()
        self.ner = ner_model
        self.new_entities = []
        self.entity_lookup = []
        self.entity_mapping = []

    def replace_word(self, text: str, word1: str, word2: str):
        """
        Find and replace strings in a given piece of text.

        Args:
            text (str): A string containing a target substring word1.
            word1 (str): Target to replace with word2 in text.
            word2 (str): Replacement for word1 in text.
        Returns:
            text (str): With word1 replaced with word2
        """
        pattern = r'\b' + re.escape(word1) + r'\b'
        return re.sub(pattern, word2, text, count=1)

    def format_align_digits(self, text, reference_text):
        if len(text) != len(reference_text):
            for idx, t in enumerate(reference_text):
                if not t.isdigit():
                    text = text[:idx] + reference_text[idx] + text[idx:]
        return text

    def fpe_encrypt(self, value: str):
        """
        Encrypt value using FPE
        """
        return self.format_align_digits(
            self.cipher_fn.encrypt(
            str(value).replace("$","").replace(",","").replace(".","").replace(" ","")
            ),
            str(value)
        )

    def fpe_decrypt(self, value: str):
        """
        Decrypt FPE value
        """
        return self.format_align_digits(
            self.cipher_fn.decrypt(
                str(value).replace("$","").replace(",","").replace(".","").replace(" ","")
            ),
            str(value)
        )
    
    def M_epsilon(self, x: int, n_lower: int, n_upper: int, epsilon: float, discretization_size=100) -> int:
        """
        m-LDP for noising numerical values between bounds.

        Args:
            x (int): Value to noise with m-LDP.
            n_lower (int): Lower bound for noising x.
            n_upper (int): Upper bound for noising x.
            epsilon (float): Privacy budget.
            discretization_size: For sampling.
        Returns:
            noised_output (int): The noised output.
        """
        n_upper = int(n_upper)
        n_lower = int(n_lower)
        total_range = n_upper-n_lower
        x = (x-n_lower)*discretization_size/total_range
        p_i = []
        for s in range(discretization_size):
            p_i.append(math.exp(-abs(x-s)*epsilon/2))
        p_i = [val/sum(p_i) for val in p_i]
        noised_output = np.random.choice(range(discretization_size),1,p=p_i)*total_range/discretization_size+n_lower
        return int(noised_output[0])
    
    def encrypt_names(self, inputs: List[str], **kwargs: Dict[str, Any]) -> Union[List[List[str]], Union[List[str],List[str]], Dict[str,str]]:
        """
        FPE encryption for names of people. 

        Args:
            inputs (List[str]): List of strings with sensitive values.
        Returns:
            Union[List[str], Union[List[str],List[str]], Dict[str,str]]: Containing:
            1. new_entities (List[List[str]]): A nested list of strings, where each nested list contains the new encrypted entities.
            2. entity_lookup (Union[List[str],List[str]]): Two lists, with the first one having the list of first names and the second being the list of last names.
            3. entity_mapping (Dict[str,str]): Dict, maps ciphertext names to plaintext names.
        """
        new_entities = []
        entity_lookup = []
        entity_mapping = dict()
        use_fpe = kwargs['use_fpe']

        if use_fpe:
            try:
                names_dataset = load_data("preempt/names_dataset.json")
            except:
                print("Names dataset not found. Generating...")
                names_dataset_save_path = f"{os.getcwd()}/names_dataset.json"
                make_names_dataset(names_dataset_save_path)
                print(f"Saved at {names_dataset_save_path}")
                names_dataset = load_data(f"{names_dataset_save_path}")
            first_names, last_names = names_dataset["first_names"], names_dataset["last_names"]

            # Suppose we have n names. Chuck the last k names in the list and plug in these.
            input_first_names = []
            input_last_names = []
            for i, k_input in enumerate(inputs):
                temp_fnames = []
                temp_lnames = []
                for input in k_input:
                    # print(repr(input))
                    if " " in input:
                        temp_ab = input.split()
                        a, bs = temp_ab[0], temp_ab[1:]
                        input_first_names.append(a)
                        temp_fnames.append(a)
                        for b in bs:
                            input_last_names.append(b)
                            temp_lnames.append(b)
                    else:
                        input_first_names.append(input)
                        temp_fnames.append(input)
                
                # For cases where the first or last name of a
                # full name appear independently.
                # The full name will always be modified first,
                # so there is no scope of 2 word name becoming
                # a 4 word name.
                for tt in temp_fnames:
                    inputs[i].append(tt)
                for tt in temp_lnames:
                    inputs[i].append(tt)
            # Get rid of first names like Jean Baptiste
            temp = []
            for name in first_names:
                if " " not in name:
                    temp.append(name)

            first_names = temp[:]

            temp = []
            for name in last_names:
                if " " not in name:
                    temp.append(name)
            last_names = temp[:]

            # Get rid of repetitions in the list itself.
            excess_fnames = 0
            excess_lnames = 0
            for name in input_first_names:
                if name in first_names:
                    first_names.remove(name)
                    excess_fnames+=1
            for name in input_last_names:
                if name in last_names:
                    last_names.remove(name)
                    excess_lnames+=1

            first_names = first_names[:1000]
            last_names = last_names[:1000]
            first_names = first_names[:min(-len(input_first_names)-1,-2)] + list(input_first_names) + [first_names[-1]]
            last_names = last_names[:min(-len(input_last_names)-1,-2)] + list(input_last_names) + [last_names[-1]]
            # assert len(first_names)==1000
            # assert len(last_names)==1000

        offset = 3
        for k_input in inputs:
            temp = []
            for k, input in enumerate(k_input):
                if use_fpe:
                    if " " in input:
                        # If already first-last name, run and conjoin.
                        t_names = input.split()
                        a = t_names[0]
                        pt_fn_idx = first_names.index(a)
                        pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                        pt_idx = "" + pt_fn_idx

                        for b in t_names[1:]:
                            pt_ln_idx = last_names.index(b)
                            pt_ln_idx = "0"*(offset-len(str(pt_ln_idx))) + str(pt_ln_idx)
                            pt_idx += pt_ln_idx

                        ct_idx = self.fpe_encrypt(pt_idx)
                        ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]

                        ct_name = first_names[int(ct_idxs[0])]
                        for b in ct_idxs[1:]:
                            ct_last_name = last_names[int(b)]
                            ct_name = ct_name + " " + ct_last_name

                        if ct_name not in temp: temp.append(ct_name)
                    else:
                        a = input
                        try:
                            pt_fn_idx = first_names.index(a)
                            pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                            pt_idx = "" + pt_fn_idx + "9"*offset

                            ct_idx = self.fpe_encrypt(pt_idx)
                            ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]
                            ct_name = first_names[int(ct_idxs[0])]
                        except:
                            pt_fn_idx = last_names.index(a)
                            pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                            pt_idx = "" + "9"*offset + pt_fn_idx

                            ct_idx = self.fpe_encrypt(pt_idx)
                            ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]
                            ct_name = last_names[int(ct_idxs[0])]
                        
                        for b in ct_idxs[1:]:
                            ct_last_name = last_names[int(b)]
                            ct_name = ct_name + " " + ct_last_name

                        if ct_name not in temp: temp.append(ct_name)

                    # Grad mapping for decoding and for sanity.
                    entity_mapping[ct_name] = input

                else:
                    first_names = []
                    last_names = []
                    temp_name = names.get_full_name(gender='male')
                    temp.append(temp_name)
                    entity_mapping[temp_name] = input

            new_entities.append(temp)
        entity_lookup.append(first_names)
        entity_lookup.append(last_names)

        return new_entities, entity_lookup, entity_mapping

    def decrypt_names(self, inputs: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        """
        Desanitizes names in a list of sanitized strings.

        Args:
            inputs (List[str]): List of sanitized strings
        Returns:
            decrypted_lines (List[str]): List of desanitized strings
        """       
        def check(name_list, target):
            for i, n in enumerate(name_list):
                if finder(name_list[i], target):
                    return i
        def finder(word1, word2):
            encoded1 = unicodedata.normalize('NFC', word1)
            encoded2 = unicodedata.normalize('NFC', word2)
            return encoded1==encoded2

        extraction = kwargs['extraction']
        use_cached_values = kwargs['use_cache']
        decrypted_lines = []

        for line_idx, line in enumerate(inputs):
            offset = 3
            first_names, last_names = self.entity_lookup
            if extraction is None or use_cached_values:
                decrypt_target = self.entity_mapping
            else:
                decrypt_target = extraction
            
            # print(decrypt_target)
            # print("FIRST NAMES", first_names)
            for i, name in enumerate(decrypt_target[line_idx]):
                try:
                    if " " in name and name not in first_names:
                        # print("HERE")
                        t_names = name.split()
                        a = t_names[0]
                        pt_fn_idx = check(first_names, a)

                        pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                        pt_idx = "" + pt_fn_idx

                        for b in t_names[1:]:
                            pt_ln_idx = check(last_names, b)
                            pt_ln_idx = "0"*(offset-len(str(pt_ln_idx))) + str(pt_ln_idx)
                            pt_idx += pt_ln_idx

                        ct_idx = self.fpe_decrypt(pt_idx)
                        ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]

                        # print(ct_idxs)
                        ct_name = first_names[int(ct_idxs[0])]
                        for b in ct_idxs[1:]:
                            if b=="9"*offset:
                                continue
                            ct_last_name = last_names[int(b)]
                            ct_name = ct_name + " " + ct_last_name

                        pt_name = ct_name

                    else:
                        a = name
                        pt_fn_idx = first_names.index(a)
                        pt_fn_idx = "0"*(offset-len(str(pt_fn_idx))) + str(pt_fn_idx)
                        pt_idx = "" + pt_fn_idx + "9"*offset

                        ct_idx = self.fpe_decrypt(pt_idx)
                        ct_idxs = [ct_idx[i:i+offset] for i in range(0, len(ct_idx), offset)]

                        ct_name = first_names[int(ct_idxs[0])]
                        for b in ct_idxs[1:]:
                            ct_last_name = last_names[int(b)]
                            ct_name = ct_name + " " + ct_last_name            

                        pt_name = ct_name

                except:
                    raise Exception("Decrypt target not found in names dataset!")
            
                # print(name, pt_name)
                # print(line)
                line = line.replace(name, pt_name)
                # print(line)
            decrypted_lines.append(line)

        return decrypted_lines

    def encypt_money(self, inputs: List[str], **kwargs: Dict[str, Any]) -> Union[List[List[str]], Union[List[str],List[str]], List[Dict[str,List[str]]]]:
        """
        Encrypting numerical money values with FPE or m-LDP

        Args:
            inputs (List[str]): List of strings with sensitive values.
        Returns:
            Union[List[List[str]], Union[List[str],List[str]], Dict[str,str]]: Containing:
            1. new_entities (List[List[str]]): A nested list of strings, where each nested list contains the new encrypted entities.
            2. entity_lookup (Union[List[str],List[str]]): A nested list of plaintext sensitive attributes.
            3. entity_mapping (List[Dict[str,List[str]]]): A list, contains dicts with a list of ciphertext and corresponding plaintext values.
        """
        new_entities = []
        entity_lookup = []
        entity_mapping = []
        use_fpe, use_mdp = kwargs['use_fpe'], kwargs['use_mdp']
        epsilon = kwargs['epsilon']

        valid_indices = []
        for kk, input in enumerate(inputs):
            temp = []
            text_pt = []
            temp_dict = {}
            trip = 0
            for real_money in input:
                try:
                    if use_fpe:
                        offset = 6
                        val = "9"*offset + str(real_money)
                        money = self.fpe_encrypt(val)

                    elif use_mdp:
                        if int(float(real_money))==1:
                            money = round(self.M_epsilon(int(float(real_money)),1,2,epsilon), 2)
                        elif int(float(real_money))<100:
                            money = round(self.M_epsilon(int(float(real_money)),2,1000,epsilon), 2)
                        elif int(float(real_money))<1000:
                            money = round(self.M_epsilon(int(float(real_money)),100,10000,epsilon), 2)
                        else:
                            money = round(self.M_epsilon(int(float(real_money)),1000,10000,epsilon), 2)

                    temp.append(str(money))
                    text_pt.append(str(real_money))
                    temp_dict[str(money)] = str(real_money)
                except Exception as e:
                    print(e)
                    trip += 1
                    print("Error value:", real_money)
                    temp.append('None')

            if trip==0: valid_indices.append(kk)
            entity_lookup.append(text_pt)
            new_entities.append(temp)
            entity_mapping.append(temp_dict)

        return new_entities, entity_lookup, entity_mapping

    def decrypt_money(self, inputs: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        """
        Desanitizes currency values in a list of sanitized strings.

        Args:
            inputs (List[str]): List of sanitized strings
        Returns:
            decrypted_lines (List[str]): List of desanitized strings
        """
        decrypted_lines = []
        use_fpe = kwargs['use_fpe']
        use_mdp = kwargs['use_mdp']
        extraction = kwargs['extraction']
        use_cached_values = kwargs['use_cache']
        if extraction is None or use_cached_values:
            decrypt_target = self.entity_mapping
        else:
            decrypt_target = extraction
        if use_fpe:
            offset=7
            for line_idx, line in enumerate(inputs):
                for value in decrypt_target[line_idx]:
                    val = str(value)
                    if len(val)<6: continue
                    decrypt = self.fpe_decrypt(val)
                    decrypt = decrypt[6:]
                    if value==None: continue
                    if value in line:
                        line = line.replace(value, decrypt)
                    elif value.replace(".", ",") in line:
                        line = line.replace(value.replace(".", ","), decrypt.replace(".", ","))
                    else:
                        line = line.replace(value.replace(".", ","), decrypt.replace(".", ","))
                decrypted_lines.append(line)

        elif use_mdp:
            for line_idx, line in enumerate(inputs):
                for value in decrypt_target[line_idx]:
                    line = self.replace_word(line, value, self.entity_mapping[line_idx][value])
                decrypted_lines.append(line)

        return decrypted_lines

    def encrypt_age(self, inputs: List[str], **kwargs)-> Union[List[List[str]], Union[List[str],List[str]], List[Dict[str,List[str]]]]:
        """
        Encrypting age values with m-LDP

        Args:
            inputs (List[str]): List of strings with sensitive values.
        Returns:
            Union[List[List[str]], Union[List[str],List[str]], Dict[str,str]]: Containing:
            1. new_entities (List[List[str]]): A nested list of strings, where each nested list contains the new encrypted entities.
            2. entity_lookup (Union[List[str],List[str]]): A nested list of plaintext sensitive attributes.
            3. entity_mapping (List[Dict[str,List[str]]]): A list, contains dicts with a list of ciphertext and corresponding plaintext values.
        """
        new_entities = []
        entity_lookup = []
        entity_mapping = []
        epsilon = kwargs['epsilon']
        for input in inputs:
            temp = []
            text_pt = []
            temp_dict = {"pt":[],"ct":[]}
            for real_age in input:
                try:
                    text_pt.append(real_age)
                    temp_age = self.M_epsilon(int(real_age),10,99,epsilon)
                    age = str(temp_age)
                    temp.append(age)
                    temp_dict['ct'].append(age)
                    temp_dict['pt'].append(str(real_age))
                except:
                    temp.append(None)
            entity_mapping.append(temp_dict)
            entity_lookup.append(text_pt)
            new_entities.append(temp)

        return new_entities, entity_lookup, entity_mapping

    def decrypt_age(self, inputs: List[str], **kwargs) -> List[str]:
        """
        Desanitizes age values in a list of sanitized strings.

        Args:
            inputs (List[str]): List of sanitized strings
        Returns:
            decrypted_lines (List[str]): List of desanitized strings
        """
        decrypted_lines = []
        # extraction = kwargs['extraction']
        # use_cached_values = kwargs['use_cache']
        decrypt_target = self.entity_mapping
        for line_idx, line in enumerate(inputs):
                extr = decrypt_target[line_idx]
                for k in range(len(extr['pt'])):
                    line = self.replace_word(line, extr['ct'][k], extr['pt'][k])
                decrypted_lines.append(line)

        return decrypted_lines

    def encrypt(self, inputs: List[str], epsilon=0.1, entity='Name', use_mdp=False, use_fpe=True) -> Union[List[str], List[int]]:
        """
        Takes in a list of inputs and returns a list of sanitized outputs.

        Args:
            inputs (List[str]): List of inputs with sensitive attributes.
            epsilon (float): For m-LDP.
            entity (str): Sensitive attribute to sanitize. Pick from {Name/Money/Age}.
            use_mdp (bool): Use m-LDP for encrypting numerical values.
            use_fpe (bool): Use FPE for encrypting alphanumerical values.
        Returns:
            data_encrypted (List[str]): List of sanitized strings.
            invalid_indices (List[int]): List of indices where nothing was found for sanitization.
        """
        enc_fn_mapping = {
            "Name": self.encrypt_names,
            "Money": self.encypt_money,
            "Age": self.encrypt_age,
        }
        data_encrypted = []
        invalid_indices = []
        extracted = self.ner.extract(inputs, entity_type=entity)[entity]
        self.new_entities, self.entity_lookup, self.entity_mapping = enc_fn_mapping[entity](extracted, use_fpe=use_fpe, use_mdp=use_mdp, epsilon=epsilon)

        # print(inputs)
        for i, line in enumerate(inputs):
        # Get extracted/encrypted data  for the ith line.
        # Substitute all values.
            for value, encrypt in zip(extracted[i], self.new_entities[i]):
                if value is not None and encrypt is not None:
                    line = unicodedata.normalize('NFC',line).replace(unicodedata.normalize('NFC',value), encrypt)
                else:
                    invalid_indices.append(i)
                    break

            data_encrypted.append(line)

        return data_encrypted, invalid_indices
    
    def decrypt(
            self, 
            inputs: List[str], 
            entity='Name', 
            extracted: Optional[Dict[str, List[str]]]=None, 
            use_mdp=False, use_fpe=True, use_cache=False,
        ):
        """
        Takes in a list of inputs and returns a list of desanitized outputs.
        Encrypt must be used before this method!

        Args:   
            inputs (List[str]): List of sanitized inputs.
            extracted (Optional[Dict[str, List[str]]]): Dictionary of extracted sensitive attributes (see NER.extract())
            use_mdp (bool): Retrieve cached values during decryption.
            use_fpe (bool): Use FPE for decrypting alphanumerical values.
            use_cache (bool): Use cipher text values cached during santization, instead of using freshly extracted values (when NER is not reliable).
        """
        dec_fn_mapping = {
            "Name": self.decrypt_names,
            "Money": self.decrypt_money,
            "Age": self.decrypt_age,
        }
        data_decrypted = []
        if extracted is None: extracted = self.ner.extract(inputs, entity_type=entity)[entity]
        data_decrypted = dec_fn_mapping[entity](inputs, extraction=extracted, 
                                                use_fpe=use_fpe, use_mdp=use_mdp, use_cache=use_cache)

        return data_decrypted
