import os
from coref_resolver import Resolver


genre = "nw" # Other options: https://natural-language-understanding.fandom.com/wiki/OntoNotes
model_name = "spanbert_base" # The fine-tuned model to use. Options are: bert_base, spanbert_base, bert_large, spanbert_large


# needed for tensorflow model configuration setting
os.environ['data_dir'] = "coref"
os.environ['model_name'] = model_name
os.environ['GPU'] = '-1'

# coref resolver that loads model
resolver = Resolver(genre, model_name)

# text fot coref resolution
text = "Deepika has a dog. She loves him. The movie star has always been fond of animals."

# call resolve method as many without loading model again
resolved = resolver.resolve(text)

print(resolved)