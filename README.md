# preempt
This is a modular version of Prϵϵmpt, meant to be used as part of other projects. 

For the experiments and results found in [Prϵϵmpt: Sanitizing Sensitive Prompts for LLMs](https://arxiv.org/abs/2504.05147), please refer to [this repo](https://github.com/danshumaan/preempt-experiments).
## Setup
1. Clone this repo and navigate to the root directory (`preempt`).
2. Install uv following the [instructions here](https://docs.astral.sh/uv/getting-started/installation/).
3. Create a virtual environment with Python 3.11 and activate it:
```
uv venv --python 3.11
. ./.venv/bin/activate
uv sync
```


## Usage
Additional usage examples can be found in `demo.ipynb`.

We will add support for generalized NER and sanitization in the near future. 

### Complete Usage Example
This is a complete usage example where we sanitize names and currency values. Make sure you either have [Universal NER](https://huggingface.co/Universal-NER/UniNER-7B-all) or [Llama-3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) available. 

1. Import all utilities:
```
# Import utils
from preempt.utils import *
```

2. Initialize a `NER` and `Sanitizer` object:
```
# Load NER object
# ner_model = NER("/path/to/uniner-7b-pii-v3", device="cuda:1")
ner_model = NER("/path/to/Meta-Llama-3-8B-Instruct/", device="cuda:1")

# Load Sanitizer object
sanitizer_name = Sanitizer(ner_model, key = "EF4359D8D580AA4F7F036D6F04FC6A94", tweak = "D8E7920AFA330A73")
sanitizer_money = Sanitizer(ner_model, key = "FF4359D8D580AA4F7F036D6F04FC6A94", tweak = "E8E7920AFA330A73")

# Sentences
sentences = ["Ben Parker and John Doe went to the bank and withdrew $200.", "Adam won $20 in the lottery."]
```

3. Sanitize names in `sentences`:
```
# Sanitizing names
sanitized_sentences, _ = sanitizer_name.encrypt(sentences, entity='Name', epsilon=1)
print("Sanitized sentences:")
print(sanitized_sentences)
"""
Prints:

Sanitized sentences:
['Jay Francois and Lamine Franklin went to the bank and withdrew $200.', 'Elie Vinod won $20 in the lottery.']
"""
```

4. Sanitize currency values in `sanitized_sentences`:
```
# Sanitizing currency values
sanitized_sentences, _ = sanitizer_money.encrypt(sanitized_sentences, entity='Money', epsilon=1)
print("Sanitized sentences:")
print(sanitized_sentences)
"""
Prints:

Sanitized sentences:
['Jay Francois and Lamine Franklin went to the bank and withdrew $769451698.', 'Elie Vinod won $37083668 in the lottery.']
"""
```

5. Desanitize encrypted names in `sanitized_sentences`:
```
# Desanitizing names
desanitized_sentences = sanitizer_name.decrypt(sanitized_sentences, entity='Name')
print("Desanitized sentences:")
print(desanitized_sentences)
"""
Prints:

Desanitized sentences:
['Ben Parker and John Doe went to the bank and withdrew $769451698.', 'Adam won $37083668 in the lottery.']
"""
```

6. Desanitize encrypted currency values in `desanitized_sentences`:
```
# Desanitizing currency values
desanitized_sentences = sanitizer_money.decrypt(desanitized_sentences, entity='Money')
print("Desanitized sentences:")
print(desanitized_sentences)
"""
Prints:

Desanitized sentences:
['Ben Parker and John Doe went to the bank and withdrew $200.', 'Adam won $20 in the lottery.']
"""
```

### Extraction
We currently support [Universal NER](https://huggingface.co/Universal-NER/UniNER-7B-all) and [Llama-3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) for NER. We will add support for including your own NER models in the near future. 

Initialize a `NER` class object by passing the path to one of the supported NER models mentioned above:
```
ner_model = NER("/path/to/Meta-Llama-3-8B-Instruct/", device="cuda:0")
```
Extract PII values found in a list of target strings using `ner_model.extract()`:
```
sentences = ["Ben Parker and John Doe went to the bank.", "Who was late today? Adam."]
extracted = ner_model.extract(sentences, entity_type='{Name/Money/Age}')
```

### Sanitization
We currently only support sanitization for names, currency values and age, using either FPE or m-LDP.

Initialize a `Sanitizer` class object by passing the previously initialized `ner_model`, a `key` and `tweak` parameter (required for the FF3 cipher used for FPE).
```
sanitizer = Sanitizer(ner_model, key = "EF4359D8D580AA4F7F036D6F04FC6A94", tweak = "D8E7920AFA330A73")
```
Sanitize a list of target strings using `sanitizer.encrypt()`:
```
sanitized_sentences, _ = sanitizer.encrypt(sentences, entity='Name', epsilon=1, use_fpe=True, use_mdp=False)
```
PII values found during NER are stored under `sanitizer.new_entities` as a nested list.

The mappings between plain text and cipher text PII values are stored under `sanitizer.entity_mapping`. FPE will typically extract PII values from the sanitized sentences before decryption.

Sanitized sentences can be desanitized using `sanitizer.decrypt()`:
```
desanitized_sentences = sanitizer.decrypt(sanitized_sentences, entity='Name')
```

#### Sanitizing multiple PII attributes
If you want to sanitize multiple sensitive attributes, create a sanitizer for each category separately. 

For more examples, check out `demo.ipynb`

### Usage tips
NER typically works better when the inputs are smaller. Consider breaking a large chunk of text into smaller sentences when using the sanitizer.
