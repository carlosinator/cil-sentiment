import tensorflow as tf
import transformers
import tensorflow_probability as tfp
from transformers import AutoTokenizer, TFAutoModel, AutoConfig, TFAutoModelForSequenceClassification

"""
  MINION
"""
class MINION(tf.keras.Model):
  def __init__(self, model_name, num_labels, hidden_dimension=32):
    super().__init__()
    self.num_labels = num_labels
    self.hidden_dimension = hidden_dimension
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    self.encoder = TFAutoModel.from_pretrained(model_name, config=config)
    self.encoder.trainable = False
    self.linear1 = tf.keras.layers.Dense(2 * hidden_dimension)
    self.linear2 = tf.keras.layers.Dense(config.hidden_size, activation='tanh')
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hidden_dimension, return_sequences=True, return_state=False))
    self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')

  def call(self, input):
    hidden_encoder_states = self.encoder(input, return_dict=True).hidden_states

    correction = 0 # embedding layer is assumed to have correction 0
    for i in range(len(hidden_encoder_states)):
      correction = self.gru(self.linear1(hidden_encoder_states[i]) + correction)

    corrected_last_state = hidden_encoder_states[-1] + self.linear2(correction)

    return self.classifier(tf.keras.layers.GlobalAveragePooling1D()(corrected_last_state))

"""
  READ with RNN
"""
class READ(tf.keras.Model):
  def __init__(self, model_name, num_labels, hidden_dimension=32):
    super().__init__()
    self.num_labels = num_labels
    self.hidden_dimension = hidden_dimension
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    self.encoder = TFAutoModel.from_pretrained(model_name, config=config)
    self.encoder.trainable = False
    self.linear1 = tf.keras.layers.Dense(self.hidden_dimension)
    self.linear2 = tf.keras.layers.Dense(self.hidden_dimension, activation='tanh')
    self.linear3 = tf.keras.layers.Dense(config.hidden_size, activation='tanh')
    self.classifier = tf.keras.layers.Dense(self.num_labels, activation='softmax')

  def call(self, input):
    hidden_encoder_states = self.encoder(input, return_dict=True).hidden_states

    correction = 0 # embedding layer is assumed to have correction 0
    for i in range(len(hidden_encoder_states)):
      correction = self.linear2(self.linear1(hidden_encoder_states[i]) + correction)

    corrected_last_state = hidden_encoder_states[-1] + self.linear3(correction)

    return self.classifier(tf.keras.layers.GlobalAveragePooling1D()(corrected_last_state))

"""
  READ with GRU
"""
class READ_GRU(tf.keras.Model):
  def __init__(self, model_name, num_labels, hidden_dimension=32):
    super().__init__()
    self.num_labels = num_labels
    self.hidden_dimension = hidden_dimension
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    self.encoder = TFAutoModel.from_pretrained(model_name, config=config)
    self.encoder.trainable = False
    self.linear = tf.keras.layers.Dense(config.hidden_size, activation='tanh')
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dimension, return_sequences=False, return_state=True))
    self.classifier = tf.keras.layers.Dense(self.num_labels, activation='softmax')

  def call(self, input):
    hidden_encoder_states = self.encoder(input, return_dict=True).hidden_states
    num_hidden_states = len(hidden_encoder_states)
    sequence_length = input['input_ids'].shape[1]

    inputs = tf.stack(hidden_encoder_states) # convert list of tensors to tensor
    # change dimensions (num_hidden_states, batchsize, sequence_length, dim) to (sequence_length, batchsize, num_hidden_states, dim)
    inputs = tf.transpose(inputs, perm=[2, 1, 0, 3])

    outputs = []
    for i in range(sequence_length):
      outputs.append(tf.keras.layers.Add()(self.gru(inputs[i])[1:]))
    last_correction = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])

    corrected_last_state = hidden_encoder_states[-1] + self.linear(last_correction)
    return self.classifier(tf.keras.layers.GlobalAveragePooling1D()(corrected_last_state))
