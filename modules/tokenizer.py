import pickle
import argparse

import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer

from utils import get_file_name, get_file_format, build_file_path

class SquadTokenizer:
    def __init__(self, path, max_input_vocab = 53000, max_output_vocab = 28000, max_input_length = 50, max_output_length = 20):
        format_ = get_file_format(path)
        if format_ == 'csv':
            self.data = pd.read_csv(path)
        elif format_ == 'pkl':
            self.data = pd.read_pickle(path)
        else:
            raise ValueError('Invalid file format. Should be csv or pkl')

        self.max_input_vocab = max_input_vocab
        self.max_output_vocab = max_output_vocab
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length


    def word_tokenize(self, data, remove_punc = True, merge_context_answer = True):
        tokenizer_func = tokenize_words_remove_punc if remove_punc else tokenize_words

        input_tokenized = tokenizer_func(data['context_answer'])
        if merge_context_answer:
            answer_tokenized = tokenizer_func(data['answer'])
            input_tokenized = [tr + ans for tr, ans in zip(input_tokenized, answer_tokenized)]
        
        question_tokenized = tokenizer_func(data['question'])

        return [input_tokenized, question_tokenized]


    def filter_by_length(self, tokenized_data):
        new_train_sequences = []
        new_question_sequences = []

        for input_seq, que_seq in zip(tokenized_data[0], tokenized_data[1]):
          if len(input_seq) > self.max_input_length or len(que_seq) > self.max_output_length:
            continue
        
          new_train_sequences.append(input_seq)
          new_question_sequences.append(que_seq)
  
        return [new_train_sequences, new_question_sequences]

    def build_tokenizers(self, tokenized_data):
        input_tokenizer = build_tokenizer(tokenized_data[0], self.max_input_vocab)
        output_tokenizer = build_tokenizer(tokenized_data[1], self.max_output_vocab)
        self.tokenizers = [input_tokenizer, output_tokenizer]

    
    def index_tokenize(self, tokenized_data, padding = 'post'):
        
        input_index_tokenized = self.tokenizers[0].texts_to_sequences(tokenized_data[0])
        output_index_tokenized = self.tokenizers[1].texts_to_sequences(tokenized_data[1])
        
        if padding in ['post', 'pre']:
            max_input_length = max([len(seq) for seq in input_index_tokenized])
            max_output_length = max([len(seq) for seq in output_index_tokenized])
            input_index_tokenized = pad_sequences(input_index_tokenized, maxlen=max_input_length, padding=padding)
            output_index_tokenized = pad_sequences(output_index_tokenized, maxlen=max_output_length, padding=padding)
        
        return  [input_index_tokenized, output_index_tokenized]


    def fit(self, remove_punc = True, merge_context_answer = True, filter_by_length = True, padding = 'post'):
        self.tokenized_data = self.word_tokenize(self.data, remove_punc, merge_context_answer)
        self.tokenized_data[1] = add_start_end_tokens(self.tokenized_data[1])
        if filter_by_length:
            self.tokenized_data = self.filter_by_length(self.tokenized_data)

        self.build_tokenizers(self.tokenized_data)
        
        return self.index_tokenize(self.tokenized_data, padding=padding)

    def texts_to_sequences(self, context_texts, question_texts, answer_texts = None, remove_punc = True, merge_context_answer = True, filter_by_length = False, padding = 'post'):
        tokenized_data = self.word_tokenize([context_texts, question_texts, answer_texts], remove_punc, merge_context_answer)
        if filter_by_length:
            tokenized_data = self.filter_by_length(tokenized_data)

        return self.index_tokenize(tokenized_data, padding=padding)


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)




def tokenize_words(sequences):
    return [wordpunct_tokenize(str(seq)) for seq in sequences]

def tokenize_words_remove_punc(sequences):
    tk = RegexpTokenizer(r'\w+')
    return [tk.tokenize(str(seq)) for seq in sequences]


def add_start_end_tokens(sequences):
  new_sequences = sequences
  for i, seq in enumerate(sequences):
    new_sequences[i] = ['<sos>'] + seq + ['<eos>']
  
  return new_sequences


def build_tokenizer(sequences, max_words):
    tok = Tokenizer(num_words=max_words, oov_token='<unk>')
    tok.fit_on_texts(sequences)
    return tok



def parse_arguments():
    parser = argparse.ArgumentParser(description='Builds tokenizers given the inputs and saves the tokenizer object.')
    parser.add_argument('--input', type=str, help='Path for the squad csv or pkl file.', required=True)
    parser.add_argument('--out', type=str, help='Output path for tokenizer object.', default='./tokenizer.pkl')
    parser.add_argument('--data_out', type=str, help='Output for the tokenized data', default='./train_data.pkl')
    parser.add_argument('--padding', type=str, help='Padding mode.', choices=['pre', 'post'], default='post')
    parser.add_argument('--include_answers', type=bool, help='Including answers in the input or not', default=True)
    parser.add_argument('--filter_by_length', type=bool, help='Filter out the observations less than the given max_input_length or max_output_length', default=True)
    parser.add_argument('--max_input_length', type=int, help='Max input length', default=50)
    parser.add_argument('--max_output_length', type=int, help='Max output length', default=20)
    parser.add_argument('--max_input_vocab', type=int, help='Max input vocab size. Less frequent words are replaced with <unk>', default=53000)
    parser.add_argument('--max_output_vocab', type=int, help='Max output vocab size. Less frequent words are replaced with <unk>', default=28000)
    parser.add_argument('--remove_punc', type=bool, help='Remove punctuations during tokenizer or not', default=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    tokenizer = SquadTokenizer(args.input, args.max_input_vocab, args.max_output_vocab, args.max_input_length, args.max_output_length)
    tokenized_data = tokenizer.fit(args.remove_punc, args.include_answers, args.filter_by_length, args.padding)
    
    with open(args.data_out, 'wb') as f:
        pickle.dump(tokenized_data, f, pickle.HIGHEST_PROTOCOL)

    tokenizer.save(args.out)
