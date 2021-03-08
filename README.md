### SpanBERT Coref Resolver
This repository is just a wrapper on top of [coref](https://github.com/mandarjoshi90/coref) and with the help of this [notebook] (https://colab.research.google.com/drive/1SlERO9Uc9541qv6yH26LJz5IM9j7YVra#scrollTo=H0xPknceFORt). The additional work is the replacement of the mentions.

#### How to use

##### Installation

```
git clone --recurse-submodules https://github.com/sayef/bert-coref-resolver.git
cd bert-coref-resolver/coref
pip install -r requirements.txt
./setup_all.sh
./download_pretrained.sh spanbert_base models
```

##### Usage

1. 

2. Python demo file:

```
import os
import nltk
nltk.download('punkt')
import sys
import warnings
import contextlib

from resolver import Resolver
    

genre = "nw" # Other options: https://natural-language-understanding.fandom.com/wiki/OntoNotes
model_name = "spanbert_base" # The fine-tuned model to use. Options are: bert_base, spanbert_base, bert_large, spanbert_large
model_dir = os.path.join("./coref/models") # where you have extracted pretrained model, if possible give absolute path

# coref resolver that loads model
coref_resolver = Resolver(genre, model_dir, model_name)
    
# text fot coref resolution
text = "Deepika has a dog. She loves him. The movie star has always been fond of animals."

# call resolve method as many without loading model again


resolved = coref_resolver.resolve(text)

print(resolved)
```

Output: `Deepika has a dog. Deepika loves a dog. Deepika has always been fond of animals.`
