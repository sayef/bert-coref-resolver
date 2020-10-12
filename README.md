### SpanBERT Coref Resolver
This repository is just a wrapper on top of [coref](https://github.com/mandarjoshi90/coref) and with the help of this [notebook] (https://colab.research.google.com/drive/1SlERO9Uc9541qv6yH26LJz5IM9j7YVra#scrollTo=H0xPknceFORt). The additional work is the replacement of the mentions.

#### How to use

##### Installation

```
git clone --recurse-submodules https://github.com/sayef/bert-coref-resolver.git
cd bert-coref-resolver/coref
pip install -r requirements.txt
./setup_all.sh
./download_pretrained.sh spanbert_base
```

##### Usage

1. Start by adding `coref` module to PYTHONPATH and launch your python script. 
`export PYTHONPATH='coref' python`

2. Python script / file

```
import os
from coref_resolver import Resolver


genre = "nw" # Other options: https://natural-language-understanding.fandom.com/wiki/OntoNotes
model_name = "spanbert_base" # The fine-tuned model to use. Options are: bert_base, spanbert_base, bert_large, spanbert_large


# needed for tensorflow model configuration setting
os.environ['data_dir'] = "coref"
os.environ['model_name'] = model_name
os.environ['GPU'] = '-1'

resolver = Resolver(genre, model_name)

text = "Deepika has a dog. She loves him. The movie star has always been fond of animals"

resolved = resolver.resolve(text)
print(resolved)
```

Output: `Deepika has a dog. Deepika loves a dog. Deepika has always been fond of animals.`
