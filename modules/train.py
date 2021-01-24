import json
import argparse
import pickle

import tensorflow as tf

from model import Encoder, Decoder
from embeddings import load_word_embeddings
from tokenizer import SquadTokenizer



class Trainer:
    def __init__(self, train_config_path):
        self.train_config = self.load_train_config(train_config_path)

        train_data_path = self.train_config["train_data_path"]
        val_data_path = self.train_config["val_data_path"]
        tokenizer_path = self.train_config["tokenizer_path"]
        embeddings_path = self.train_config["embeddings_path"]

        self.tokenizers = self.load_tokenizers(tokenizer_path)
        self.train_data = self.build_dataset(train_data_path)

        if val_data_path is not None:
            self.val_data = self.build_dataset(val_data_path)
        
        self.word_embeddings = load_word_embeddings(embeddings_path)

        self.build_model()
        self.define_objective()


    def load_train_config(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def load_tokenizers(self, path):
        with open(path, 'rb') as f:
            sq_tokenizer = pickle.load(f)
        
        return sq_tokenizer.tokenizers

    def build_dataset(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1])).shuffle(len(data[0]))
        dataset = dataset.batch(self.train_config['batch_size'])

        return dataset

    def build_model(self):
        self.encoder = Encoder(self.train_config["max_input_size"], 
                               self.word_embeddings, 
                               self.tokenizers[0].word_index,
                               self.train_config["hidden_unit_size"], 
                               self.train_config["batch_size"])

        self.decoder = Decoder(self.train_config["max_output_size"], 
                               self.word_embeddings, 
                               self.tokenizers[1].word_index,
                               self.train_config["hidden_unit_size"], 
                               self.train_config["batch_size"])

    def define_objective(self):
        optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.train_config["lr"], 
                    clipvalue=self.train_config["grad_clip"])

        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, 
                    reduction='none')

        self.optimizer = optimizer
        self.loss_func = loss_func
    
        return optimizer, loss_func

    def calculate_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_func(real, pred)   
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask   
        return tf.reduce_mean(loss_)


    @tf.function
    def train_step(self, inputs, targets, enc_hidden):
      loss = 0

      with tf.GradientTape() as tape:
        enc_output, hidden_states = self.encoder((inputs, enc_hidden), training=True)

        dec_hidden = hidden_states

        dec_input = tf.expand_dims([self.tokenizers[1].word_index['<sos>']] * self.train_config["batch_size"], 1)

        for t in range(1, targets.shape[1]):
          predictions, dec_hidden, _ = self.decoder((dec_input, dec_hidden, enc_output), training=True)
          loss += self.calculate_loss(targets[:, t], predictions)
          dec_input = tf.expand_dims(targets[:, t], 1)

      batch_loss = (loss / int(targets.shape[1]))
      variables = self.encoder.trainable_variables + self.decoder.trainable_variables
      gradients = tape.gradient(loss, variables)
      self.optimizer.apply_gradients(zip(gradients, variables))

      return batch_loss


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains the encoder decoder model.')
    parser.add_argument('--train_config', type=str, help='Path for training config json', default='./train_config.json')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()

    trainer = Trainer(args.train_config)


