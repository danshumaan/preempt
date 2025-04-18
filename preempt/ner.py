from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from .utils import gen_delimiters, prompt_preprocessor, postprocess_output, any2en
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

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