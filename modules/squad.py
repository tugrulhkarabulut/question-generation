import json
import pandas as pd
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
from tqdm import tqdm
import numpy as np
import re
import contractions
import argparse
from utils import get_file_name, build_file_path

class SquadProcessor:
    def __init__(self):
        self.data = None
        self.df = None

    def read_json(self, path):
        with open(path) as f:
            my_json = json.load(f)['data']
            self.data = my_json
    
    def to_dict(self):
        if self.data is None:
            print('Data is not loaded!')
            return
        
        qa_dict = { 'context': [], 'question': [], 'answer': [], 'answer_start': [] }
        for text_obj in self.data:
            for paragraph in text_obj['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    if not qa['is_impossible']:
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            answer_start = ans['answer_start']
                            qa_dict['context'].append(context)
                            qa_dict['question'].append(question)
                            qa_dict['answer'].append(answer)
                            qa_dict['answer_start'].append(answer_start)
        
        return qa_dict
    
    def to_df(self):
        if self.data is None:
            print('Data is not loaded!')
            return

        return pd.DataFrame(self.to_dict())

    
    def find_answer_sentence(self, context, answer_start):
        answer_sentence_list = [context[answer_start]]

        i, j = answer_start - 1, answer_start + 1

        while True:
          if i < 0 or (context[i] == ' ' and context[i-1] == '.'):
            break
          answer_sentence_list.insert(0, context[i])
          i = i - 1

        while True:
          if j >= len(context) - 1 or (context[j] == '.' and context[j+1] == ' '):
            break
          answer_sentence_list.append(context[j])
          j = j + 1

        return ''.join(answer_sentence_list)
    
    def preprocess(self):
        df = self.to_df()

        df['context_answer'] = df.apply(lambda row: self.find_answer_sentence(row.context, row.answer_start), axis = 1)

        keys = ['context_answer', 'question', 'answer']

        for key in keys:
            df[key] = self.text_preprocess(df[key])

        self.df = df[keys]
        return self.df
    


    def text_preprocess(self, sequences):
        sequences = lowercase_sequences(sequences)

        to_be_changed = {
            '°c': 'celsius',
            '°f': 'fahrenheit',
            "qu'ran": 'quran',
            '−': '-',
            'asphalt/bitumen': 'asphalt bitumen',
            'us$': 'us $',
            '°': ' degree ',
            '”': '"',
            '—': '-',
            '⁄': '/',
            '′': "'",
            '″': '"',
            '×': 'times',
            '–': '-'
        }

        to_be_deleted = ['§', '्', '\ufeff']

        sequences = replace_tokens(sequences, to_be_changed)

        sequences = delete_tokens(sequences, to_be_deleted)

        sequences = handle_num_and_punc(sequences)

        sequences = split_punctuations(sequences)

        if isinstance(sequences, pd.Series):
          sequences = sequences.apply(contractions.fix)
        else:
          sequences = [contractions.fix(seq) for seq in sequences]

        return sequences

    def save(self, path, save_format = 'pkl'):
        if self.df is None:
            print('Data is not preprocessed.')
            return

        if save_format == 'pkl':
            self.df.to_pickle(path)
        else:
            self.df.to_csv(path, index = False)


def lowercase_sequences(sequences):
    if isinstance(sequences, pd.Series):
        return sequences.str.lower()
    return [seq.lower() for seq in sequences]


def replace_tokens(sequences, to_be_replaced):
    new_seqs = sequences
    for token, new_token in to_be_replaced.items():
        if isinstance(sequences, pd.Series):
          new_seqs = new_seqs.str.replace(token, new_token)
        else:
          new_seqs = [seq.replace(token, new_token) for seq in new_seqs]
    
    return new_seqs

def delete_tokens(sequences, to_be_deleted):
    to_be_deleted_str = ''.join(to_be_deleted)
    
    f = lambda text: re.sub(r'[{}]'.format(to_be_deleted_str), '', text)

    if isinstance(sequences, pd.Series):
      return sequences.apply(f)
    
    return [f(seq) for seq in sequences]

def split_num_and_dot(seq):
    return re.sub(r'[0-9]{1,}\.(?![0-9])', lambda m: m.group(0)[:-1] + " " + ".", seq)

def split_num_and_dash(seq):
    return re.sub(r'[0-9]{4}–[0-9]{2}', lambda m: m.group(0)[:4] + " and " + m.group(0)[:2] + m.group(0)[5:], seq)

def split_num_and_currency(seq):
    curr_to_str = {
        '€': 'euro',
        '£': 'pound',
        '$': 'dollar'
    }
    res = re.sub(r'[€£$]{1}[0-9]+([\.,]{1}([0-9]{0,})?)?', lambda m: m.group(0)[1:] + " " + curr_to_str[m.group(0)[0]], seq)
    return res


def handle_num_and_punc(sequences):
    if isinstance(sequences, pd.Series):    
      return sequences.apply(split_num_and_dot).apply(split_num_and_dash).apply(split_num_and_currency)

    return [split_num_and_currency(split_num_and_dash(split_num_and_dot(seq))) for seq in sequences]

def split_quoted_text(sequences):
  f = lambda text: re.sub(r' (["\'])(?:(?=(\\?))\2.)*?\1 ', lambda m: m.group(0)[0] + " " + m.group(0)[1:-1] + " " + m.group(0)[-1], text, re.DOTALL)
    
  if isinstance(sequences, pd.Series):
    return sequences.apply(f)
    
  return [f(seq) for seq in sequences]


def split_punctuations(sequences):
    f = lambda text: re.sub(r'[\(\)\.\"\'\[\]\,%;:?!-/]{2,}', lambda m: ' '.join(list(m.group(0))), text, flags=re.DOTALL)
    if isinstance(sequences, pd.Series):
      return sequences.apply(f)
    
    return [f(seq) for seq in sequences]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocesses squad json file and saves into a csv or pandas dataframe picke file.')
    parser.add_argument('--input', type=str, help='Path for the squad json file.', required=True)
    parser.add_argument('--out', type=str, help='Output path for the preprocessed data.', default='./')
    parser.add_argument('--out_format', type=str, help='Format for saving the preprocessed data.', choices=['pkl', 'csv'], default='pkl')

    args = parser.parse_args()

    input_ = args.input
    file_name = get_file_name(input_)
    format_ = args.out_format
    output = build_file_path(args.out, file_name, format_)

    return input_, output, format_


if __name__ == '__main__':
    input_, output, format_ = parse_arguments() 

    processor = SquadProcessor()
    processor.read_json(input_)
    processor.preprocess()
    processor.save(output, format_)
