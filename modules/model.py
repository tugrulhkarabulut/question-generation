import tensorflow as tf
import numpy as np

class QBaseModel(tf.keras.Model):
  def __init__(self):
    super(QBaseModel, self).__init__()
  
  def build_pretrained_embedding_layer(self, input_shape, word_embeddings, word_to_index, trainable = True, regularizer = None):
    embedding_size = word_embeddings['the'].shape[0]
    vocab_len = len(word_to_index) + 1

    initializer = tf.keras.initializers.GlorotUniform()
    
    embedding_matrix = np.zeros((vocab_len, embedding_size))
    
    for word, idx in word_to_index.items():
        if word in word_embeddings:
          embedding_matrix[idx, :] = word_embeddings[word]
        else:
          embedding_matrix[idx, :] = initializer((embedding_size,))
        
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=embedding_size, trainable = trainable, mask_zero = True, embeddings_regularizer = regularizer)

    embedding_layer.build((None, input_shape))
    
    embedding_layer.set_weights([embedding_matrix])
    
    return embedding_layer


class Encoder(QBaseModel):
  def __init__(self, max_input_size, word_embeddings, word_to_index, enc_units, batch_size):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.enc_units = enc_units
    self.embedding = self.build_pretrained_embedding_layer(max_input_size, word_embeddings, word_to_index, trainable = False)
    self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_dropout=0.2,
                                   dropout=0.1,
                                   recurrent_initializer='glorot_uniform'))

  def call(self, inputs, training = False):
    x, hidden = inputs
    
    x = self.embedding(x)

    output, state_h_fwd, state_h_bwd, state_c_fwd, state_c_bwd = self.rnn(x, initial_state = hidden, training = training)

    return output, [state_h_fwd, state_h_bwd, state_c_fwd, state_c_bwd]

  def initialize_hidden_states(self, batch_size = None):
    if batch_size is None:
      batch_size = self.batch_size

    return tf.zeros((batch_size, self.enc_units))


class Decoder(QBaseModel):
  def __init__(self, max_input_size, word_embeddings, word_to_index, dec_units, batch_size):
    super(Decoder, self).__init__()
    self.batch_size = batch_size
    self.dec_units = dec_units
    self.embedding = self.build_pretrained_embedding_layer(max_input_size, word_embeddings, word_to_index, trainable = False)
    self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform'))
    
    self.att_dropout = tf.keras.layers.Dropout(0.1)
    self.fc = tf.keras.layers.Dense(len(word_to_index))
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, inputs, training = True):
    x, hidden, enc_output = inputs

    hidden_att = tf.concat([hidden[0], hidden[2]], axis = 1)

    context_vector, attention_weights = self.attention(hidden_att, enc_output)

    context_vector = self.att_dropout(context_vector, training = training)

    x = self.embedding(x)

    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    output, state_h_fwd, state_h_bwd, state_c_fwd, state_c_bwd = self.rnn(x, initial_state = hidden, training = training)

    output = tf.reshape(output, (-1, output.shape[2]))

    x = self.fc(output)

    return x, [state_h_fwd, state_h_bwd, state_c_fwd, state_c_bwd], attention_weights



class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    self.W2 = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)

    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights