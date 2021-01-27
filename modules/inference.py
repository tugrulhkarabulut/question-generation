import argparse
import pickle
import csv

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tokenizer import SquadTokenizer
from utils import load_tokenizers, build_file_path, \
                  get_file_format, get_file_name, padded_input_to_sentence

class Inferer:
    def __init__(self, input_path, output_path, 
                 output_format, tokenizer_path, encoder_path, 
                 decoder_path, max_input_length, 
                 max_output_length, beam_search, beam_width, 
                 replace_unk_mode, verbose):
        self.output_path = output_path
        self.output_format = output_format
        self.load_data(input_path)
        self.tokenizers = load_tokenizers(tokenizer_path)

        self.encoder = self.load_model(encoder_path)
        self.decoder = self.load_model(decoder_path)

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_beam_search = beam_search
        self.beam_width = beam_width
        self.replace_unk_mode = replace_unk_mode
        self.verbose = verbose

    def load_data(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, list):
            data = data[0]

        self.data = data

    def tokenize_data(self):
        tokenized_data = []
        tokenized_data.append(self.tokenizers[0].texts_to_sequences(self.data))

        self.tokenized_data = tokenized_data

    def load_model(self, path):
        return tf.keras.models.load_model(path)

    def greedy_search(self, padded_input, store_attention = False):
        hidden = [self.encoder.initialize_hidden_states(batch_size = 1)] * 4
        enc_out, enc_hidden = self.encoder((padded_input, hidden))
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tokenizers[1].word_index['<sos>']], 0)

        attention = np.zeros((self.max_output_length, self.max_input_length))

        result = ''
        for t in range(self.max_output_length):
          predictions, dec_hidden, attention_weights = self.decoder((dec_input,
                                                               dec_hidden,
                                                               enc_out))
          if store_attention:
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention[t] = attention_weights.numpy()

          predicted_id = tf.argmax(predictions[0]).numpy()

          result += self.tokenizers[1].index_word[predicted_id] + ' '

          if self.tokenizers[1].index_word[predicted_id] == '<eos>':
            return result, attention, dec_hidden

          dec_input = tf.expand_dims([predicted_id], 0)

        return result, attention, dec_hidden

    def beam_search(self, padded_input, return_sequence=False):
        hidden = [self.encoder.initialize_hidden_states(batch_size=1)] * 4
        enc_out, enc_hidden = self.encoder((padded_input, hidden))
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tokenizers[1].word_index['<sos>']], 0)

        attention_all = np.zeros((self.max_output_length, self.max_input_length))

        predictions, dec_hidden, attention_weights = self.decoder((dec_input,
                                                               dec_hidden,
                                                               enc_out))
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_all[0] = attention_weights.numpy()

        first_candidates = tf.argsort(predictions[0])[:-self.beam_width-1:-1].numpy()
        queue = [[el] for el in first_candidates]
        candidate_states = [dec_hidden] * self.beam_width
        scores = list(np.log(predictions[0].numpy()[first_candidates]))
        attention_weights_all = [attention_all] * self.beam_width


        while len(queue) > 0:
          new_queue = []
          new_states = []
          new_att_weights = []
          new_scores = []

          new_candidate_found = False

          for index, candidate in enumerate(queue):
            candidate_state = candidate_states[index]
            candidate_score = scores[index]
            att_weights = attention_weights_all[index]


            if self.tokenizers[1].index_word[candidate[-1]] == '<eos>' or len(candidate) >= self.max_output_length:
                new_queue += [candidate]
                new_states += [candidate_state]
                new_scores += [candidate_score]
                new_att_weights += [att_weights]
                continue
            
            new_candidate_found = True

            dec_input = tf.expand_dims([candidate[-1]], 0)
            predictions, dec_hidden, attention_weights = self.decoder((dec_input,
                                                                candidate_state,
                                                                enc_out))


            candidates = tf.argsort(predictions[0])[:-self.beam_width-1:-1].numpy()

            attention_weights = tf.reshape(attention_weights, (-1, ))
            att_weights[len(candidate)] = attention_weights

            new_queue += [candidate + [el] for el in candidates]
            new_states += [dec_hidden for _ in range(self.beam_width)]
            new_att_weights += [att_weights for _ in range(self.beam_width)]
            new_scores += [(candidate_score + np.log(score)) / (len(candidate) ** 0.7) for i, score in enumerate(np.log(predictions[0].numpy()[candidates]))]


          if new_candidate_found is False:
            best_candidate, att_weights_best, _ = sorted(zip(queue, attention_weights_all, scores), key = lambda pair: pair[2])[-1]
            if return_sequence:
              return best_candidate, att_weights_best

            return padded_input_to_sentence(best_candidate, self.tokenizers[1]), att_weights_best


          candidate_score_selected = sorted(zip(new_queue, new_states, new_att_weights, new_scores), key = lambda pair: pair[3], reverse = True)[:self.beam_width]
          queue = []
          candidate_states = []
          scores = []
          attention_weights_all = []

          for candidate, state, att_weights, score in candidate_score_selected:
            queue.append(candidate)
            candidate_states.append(state)
            scores.append(score)
            attention_weights_all.append(att_weights)

        best_candidate, att_weights_best, _ = sorted(zip(queue, attention_weights_all, scores), key = lambda pair: pair[2])[-1]
        if return_sequence:
            return best_candidate, att_weights_best

        return padded_input_to_sentence(best_candidate, self.tokenizers[1]), att_weights_best

    def replace_unk(self, generated_question, input_text, attentions):
        new_q = []
        gq_words = generated_question.split()[:-1]

        for i, w in enumerate(gq_words):
            if w == '<unk>':
                best_att_i = np.argmax(attentions[i][:min(len(input_text), 
                                                    self.max_input_length)])
                new_q.append(input_text[best_att_i])
            else:
                new_q.append(w)

        new_q = ' '.join(new_q)
        return new_q


    def save_output(self):
        if self.output_format == 'pkl':
            with open(self.output_path, 'wb') as f:
                pickle.dump(self.output_data, f, pickle.HIGHEST_PROTOCOL)
        
        elif self.output_format == 'csv':
            with open(self.output_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(self.output_data[0], self.output_data[1]))


    def infer(self):
        self.tokenize_data()

        output_data = []

        
        for i, exp in tqdm(enumerate(self.tokenized_data), verbose = self.verbose):
            if self.use_beam_search:
                question, attentions = self.beam_search(exp)
            else:
                question, attentions = self.greedy_search(exp)

            if self.replace_unk_mode:
                question = self.replace_unk(question, self.data[i], attentions)
            
            output_data.append(question)

        self.output_data = output_data



def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate novel questions given the input sentences.')
    parser.add_argument('--input', type=str, help='Path for the input data', default='./squad_val.pkl')
    parser.add_argument('--out', type=str, help='Ouput path for the inferred data', default='./')
    parser.add_argument('--out_format', type=str, help='Save format for inferred data.', choices=['csv', 'pkl'], default='pkl')
    parser.add_argument('--tokenizer_path', type=str, help='Path for the tokenizer object', default='./tokenizer.pkl')
    parser.add_argument('--encoder_path', type=str, help='Path for the encoder model', default='./model')
    parser.add_argument('--decoder_path', type=str, help='Path for the decoder model.')
    parser.add_argument('--max_input_length', type=int, help='Maximum input length.', default=50)
    parser.add_argument('--max_output_length', type=int, help='Max output length.', default=20)
    parser.add_argument('--beam_search', type=bool, help='Enable or disable beam search.', default=True)
    parser.add_argument('--beam_width', type=int, help='Beam width', default=3)
    parser.add_argument('--replace-unk', type=int, help='Replace generated <unk> tokens with the word in the input that has highest attention', default=3)
    parser.add_argument('--verbose', type=bool, help='', default=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()

    input_ = args.input
    file_name = get_file_name(input_) + '_generated'
    format_ = args.out_format
    output = build_file_path(args.out, file_name, format_)

    inferer = Inferer(
                    input_, output, format_, args.tokenizer_path, 
                    args.encoder_path, args.decoder_path, args.max_input_length, 
                    args.max_output_length, args.beam_search, args.beam_width,
                    args.replace_unk, args.verbose)

    inferer.infer()
    inferer.save_output()

    

    
