import os
import nltk
nltk.download('punkt')
import sys


from resolver import Resolver
    

genre = "nw" # Other options: https://natural-language-understanding.fandom.com/wiki/OntoNotes
model_name = "spanbert_base" # The fine-tuned model to use. Options are: bert_base, spanbert_base, bert_large, spanbert_large
model_dir = "./coref/models" # where you have extracted pretrained model, if possible give absolute path

# coref resolver that loads model
coref_resolver = Resolver(genre, model_dir, model_name)
    
# text fot coref resolution
text = """
Police on Monday said they had arrested an Algerian man in the southern town of Bari on suspicion of belonging to the militant "Islamic State" (IS) group and involvement in the Paris attacks.

The coordinated incidents on November 13, 2015, which killed 130 and left hundreds more wounded, were France's deadliest attacks since World War II.

The perpetrators targeted people with guns and explosives at the Bataclan concert hall, as well as restaurants in eastern Paris and the vicinity of the Stade de France sports stadium.

Who is the arrested suspect?

A statement said that the man, named by the La Repubblica daily newspaper as Athmane T., was believed to have provided counterfeit documents to the gunmen and suicide bombers.

Investigators said the 36-year-old was part of an IS cell that had been operating in France and Belgium with his two brothers, according to Italian media.

He had reportedly already been in prison in Bari for carrying false documents and was set to be released in June.

The suspect is also believed to have had contact with two of the extremists involved in the Paris attacks of January 2015, who attacked a Jewish supermarket and the Charlie Hebdo newsroom.

The trial of 20 people charged over the jihadi attacks in November 2015 is expected to get underway in late 2021.
"""


text = ' '.join(text.split())
sentences = nltk.sent_tokenize(text)
sentences = list(map(lambda x: x[:-1] + ' ' + x[-1] if x.endswith(('!', '.', '?')) else x, sentences ))
    

resolved = coref_resolver.resolve(sentences)


resolved = nltk.sent_tokenize(resolved)
resolved = list(map(lambda x: x[0].capitalize() + x[1:-2] + x[-1] if x.endswith(('!', '.', '?')) else x, resolved ))
resolved = ' '.join(resolved)

print(resolved)