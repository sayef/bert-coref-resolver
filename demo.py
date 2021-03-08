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