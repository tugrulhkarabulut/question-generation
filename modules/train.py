import json
import argparse
import pickle
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import Encoder, Decoder
from embeddings import load_word_embeddings
from tokenizer import SquadTokenizer



class Trainer:
    def __init__(self, train_config_path, save_path, load_path, verbose):
        self.train_config = self.load_train_config(train_config_path)
        self.save_path = save_path
        self.load_path = load_path
        self.verbose = verbose

        train_data_path = self.train_config["train_data_path"]
        val_data_path = self.train_config["val_data_path"]
        tokenizer_path = self.train_config["tokenizer_path"]
        embeddings_path = self.train_config["embeddings_path"]

        self.tokenizers = self.load_tokenizers(tokenizer_path)
        self.train_data, self.train_data_size = self.build_dataset(train_data_path)

        if val_data_path is not None:
            self.val_data, _ = self.build_dataset(val_data_path)
        
        if verbose:
            print('Loading word embeddings...')
        self.word_embeddings = load_word_embeddings(embeddings_path)

        if load_path:
            self.load()
        else:
            self.build_model()

        self.define_objective()


    def load_train_config(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def load_tokenizers(self, path):
        if self.verbose:
            print('Loading tokenizers...')

        with open(path, 'rb') as f:
            sq_tokenizer = pickle.load(f)
        
        return sq_tokenizer.tokenizers

    def build_dataset(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1])).shuffle(len(data[0]))
        dataset = dataset.batch(self.train_config['batch_size'])

        return dataset, len(data[0])

    def build_model(self):
        if self.verbose:
            print('Building models...')

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


    def calculate_validation_loss(self, val_dataset, num_batches = 10):
      total_loss = 0.0   
      for (batch_x, batch_y) in val_dataset.take(num_batches):
        enc_hidden = [self.encoder.initialize_hidden_states(batch_x.shape[0])] * 4
        enc_output, enc_hidden  = self.encoder((batch_x, enc_hidden))
        dec_hidden = enc_hidden
        loss = 0.0

        for t in range(batch_y.shape[1] - 1):
          dec_input = tf.expand_dims(batch_y[:, t], 1)
          predictions, dec_hidden, _ = self.decoder((dec_input, dec_hidden, enc_output))
          loss += self.calculate_loss(batch_y[:, t + 1], predictions)

        batch_loss = loss / int(batch_y.shape[1])
        total_loss += batch_loss

      return total_loss / num_batches


    def train(self):
        epochs = self.train_config["epochs"]
        batch_size = self.train_config["batch_size"]

        steps_per_epoch = len(self.train_data_size) // batch_size

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
          start = time.time()

          print('Epoch {} Learning Rate: {:.6f}'.format(epoch + 1, self.optimizer.learning_rate.numpy()))

          enc_hidden = [self.encoder.initialize_hidden_states()] * 4
          total_train_loss = 0
          total_val_loss = 0

          for (batch, (inp, targ)) in enumerate(self.train_data.take(steps_per_epoch)):
            batch_loss = self.train_step(inp, targ, enc_hidden)
            total_train_loss += batch_loss

            if batch % 100 == 0:
                if self.val_data:
                    val_loss = self.calculate_validation_loss(self.val_data.shuffle(len([1])))
                    total_val_loss += val_loss
                    report_str = 'Epoch {} Batch {} Training Loss {:.4f} Validation Loss {:.4f}'.format(
                                                        epoch + 1,
                                                        batch,
                                                        batch_loss.numpy(),
                                                        val_loss)
                else:
                    report_str = 'Epoch {} Batch {} Training Loss {:.4f}'.format(
                                                        epoch + 1,
                                                        batch,
                                                        batch_loss.numpy())


                print(report_str)
            
            

          #checkpoint.save(file_prefix = checkpoint_prefix)
          self.save()



          avg_train_loss = total_train_loss / steps_per_epoch
          avg_val_loss = total_val_loss / (steps_per_epoch // 100 + 1)

          train_losses.append(avg_train_loss)
          val_losses.append(avg_val_loss)

          # Plot losses after the third epoch
          if self.verbose:
              if epoch >= 2:
                plt.close()
                plt.plot(np.arange(epoch + 1) + 1, train_losses, label = 'Train')
                if val_losses:
                    plt.plot(np.arange(epoch + 1) + 1, val_losses, label = 'Validation')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

              print('Epoch {} -> Training Loss {:.4f} Validation Loss {:.4f}'.format(epoch + 1,
                                              avg_train_loss, avg_val_loss ))

              print('Time taken for 1 epoch {:.0f} seconds\n'.format(time.time() - start))

          if epoch > 1 and avg_val_loss >= val_losses[-2]:
              loss_not_improved += 1


          if loss_not_improved >= 2:
              break
        

        print('Training finished in {} epochs'.format(epoch))

    def load(self, path = None):
        if not path:
            path = self.load_path
        self.encoder = tf.keras.models.load_model(path + '/encoder')
        self.decoder = tf.keras.models.load_model(path + '/decoder')

    def save(self, path = None):
        if not path:
            path = self.save_path
        self.encoder = self.encoder.save(path + '/encoder')
        self.decoder = self.decoder.save(path + '/decoder')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains the encoder decoder model.')
    parser.add_argument('--train_config', type=str, help='Path for training config json', default='./train_config.json')
    parser.add_argument('--save_path', type=str, help='Path for saving the model', default='./model')
    parser.add_argument('--load_path', type=str, help='Path for loading the model. Used for continue training.')
    parser.add_argument('--verbose', type=bool, help='Path for training config json', default=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()

    trainer = Trainer(args.train_config, args.save_path, args.load_path, args.verbose)


