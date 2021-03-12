from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import re
import json

# warning hadling for tensrflow
import warnings
warnings.filterwarnings('ignore')
import logging
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from coref import util
from coref.bert import tokenization
from nltk.tokenize import sent_tokenize


_dir = os.path.dirname(os.path.abspath(__file__))

class Resolver():
    def __init__(self, genre, model_dir, model_name):
        f = open(os.devnull, 'w')

        stdout = sys.stdout
        stderr = sys.stderr

        sys.stderr = f
        sys.stdout = f

        try:
            self.genre = genre
            self.model_name = model_name
            os.environ['model_name'] = model_name
            os.environ['data_dir'] = model_dir
            os.environ['GPU'] = '-1'

            self.tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(_dir, "coref/cased_config_vocab/vocab.txt"), do_lower_case=False)
            self.max_segment = None
            for line in open(os.path.join(_dir, 'coref/experiments.conf')):
                if line.startswith(model_name):
                    self.max_segment = True
                elif line.strip().startswith("max_segment_len"):
                    if self.max_segment:
                        self.max_segment = int(line.strip().split()[-1])
                        break
            self.model = util.get_model(util.initialize_from_env())
            self.session = tf.Session()
            self.model.restore(self.session)

        except Exception as e:
            sys.stdout = stdout
            sys.stderr = stderr
            raise e
        finally:
            sys.stdout = stdout
            sys.stderr = stderr

        

    def encode_input(self, text):
        data = {
            'doc_key': self.genre,
            'sentences': [["[CLS]"]],
            'speakers': [["[SPL]"]],
            'clusters': [],
            'sentence_map': [0],
            'subtoken_map': [0],
        }

        subtoken_num = 0
        for sent_num, line in enumerate(text):
            raw_tokens = line.split()
            tokens = self.tokenizer.tokenize(line)
            if len(tokens) + len(data['sentences'][-1]) >= self.max_segment:
                data['sentences'][-1].append("[SEP]")
                data['sentences'].append(["[CLS]"])
                data['speakers'][-1].append("[SPL]")
                data['speakers'].append(["[SPL]"])
                data['sentence_map'].append(sent_num - 1)
                data['subtoken_map'].append(subtoken_num - 1)
                data['sentence_map'].append(sent_num)
                data['subtoken_map'].append(subtoken_num)

            ctoken = raw_tokens[0]
            cpos = 0
            for token in tokens:
                data['sentences'][-1].append(token)
                data['speakers'][-1].append("-")
                data['sentence_map'].append(sent_num)
                data['subtoken_map'].append(subtoken_num)
                
                if token.startswith("##"):
                    token = token[2:]
                if len(ctoken) == len(token):
                    subtoken_num += 1
                    cpos += 1
                    if cpos < len(raw_tokens):
                        ctoken = raw_tokens[cpos]
                else:
                    ctoken = ctoken[len(token):]

        data['sentences'][-1].append("[SEP]")
        data['speakers'][-1].append("[SPL]")
        data['sentence_map'].append(sent_num - 1)
        data['subtoken_map'].append(subtoken_num - 1)

        return data


    def decode_output(self, output, text):
        comb_text = [word for sentence in output['sentences'] for word in sentence]

        seen = set()
        clusters = []
        tokens = []

        for cluster in output['predicted_clusters']:
            mapped = []
            for mention in cluster:
                seen.add(tuple(mention))
                start = output['subtoken_map'][mention[0]]
                end = output['subtoken_map'][mention[1]] + 1
                nmention = (start, end)
                mtext = ''.join(' '.join(comb_text[mention[0]:mention[1]+1]).split(" ##"))
                mapped.append((nmention, mtext))

            clusters.append(mapped)

        for line in text:
            tokens += line.split(" ")

        for cluster in clusters:
            for mention in cluster[1:]:
                if len(tokens[mention[0][0]]):
                    tokens[mention[0][0]] = " ".join(tokens[cluster[0][0][0]:cluster[0][0][1]])
                    for m in range(int(mention[0][0])+1, int(mention[0][1])):
                        tokens[m] = ""
                
        resolved = " ".join(tokens)
        resolved = re.sub(' +', ' ', resolved)
        return resolved


    def sentence_tokens(self, text):
        tokens_fixed = []
        for sent_num, line in enumerate(text):
            raw_tokens = line.split()
            tokens = line.split(" ")
            tokens_fixed += tokens
            
        return tokens_fixed
    

    def resolve(self, text):
        # print(text)
        example = self.encode_input(text)
        tensorized_example = self.model.tensorize_example(example, is_training=False)
        feed_dict = {i:t for i,t in zip(self.model.input_tensors, tensorized_example)}
        _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = self.session.run(self.model.predictions, feed_dict=feed_dict)
        predicted_antecedents = self.model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        example["predicted_clusters"], _ = self.model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
        example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
        example['head_scores'] = []

        resolved = self.decode_output(example, text)

        return resolved
