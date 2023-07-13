import tensorflow as tf
import transformers
import tensorflow_probability as tfp
from transformers import AutoTokenizer, TFAutoModel, AutoConfig, TFAutoModelForSequenceClassification

"""
  GRU-READ
"""
class GRUModel(tf.keras.Model):
  def __init__(self, model_name, num_labels, num_gru_units=32):
    super().__init__()

    self.num_labels = num_labels
    self.num_gru_units = num_gru_units

    self.encoder = TFAutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name, output_hidden_states=True))
    self.encoder.trainable = False

    self.norm = tf.keras.layers.LayerNormalization()
    self.linear = tf.keras.layers.Dense(2 * num_gru_units) # joiner network
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_gru_units, return_sequences=True, return_state=True))

    self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')

  def call(self, input, training):
    xs = self.encoder(input, return_dict=True).hidden_states
    batchsize = xs[0].shape[0]
    h = None
    sequence = tf.zeros(shape=(batchsize, xs[0].shape[1], 2 * self.num_gru_units))
    for i in range(len(xs)):
      sequence, h1, h2 = self.gru(self.norm(tf.keras.layers.Add()([self.linear(xs[i]), sequence])), initial_state=h)
      h = [h1, h2] # h = final state

    return self.classifier(tf.reshape(tf.stack(h, axis=-1), shape=[batchsize, -1]))

"""
  GRU-READ with MC-Dropout
"""
class GRUMCModel(tf.keras.Model):
  def __init__(self, model_name, num_labels, num_gru_units=32, num_forward_passes=100):
    super().__init__()

    self.num_labels = num_labels
    self.num_forward_passes = num_forward_passes
    self.num_gru_units = num_gru_units

    self.encoder = TFAutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name, output_hidden_states=True))
    self.encoder.trainable = False

    self.norm = tf.keras.layers.LayerNormalization()
    self.dropout = tf.keras.layers.Dropout(0.1)

    self.linear = tf.keras.layers.Dense(2 * num_gru_units) # joiner network
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=num_gru_units,
                                                                 dropout=0.1, recurrent_dropout=0.1,
                                                                 return_sequences=True, return_state=True))

    self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')

  def _forward(self, xs):
    h = None
    sequence = tf.zeros(shape=(xs[0].shape[0], 64))
    for i in range(len(xs)):
      sequence, h1, h2 = self.gru(self.norm(tf.keras.layers.Add()([self.linear(xs[i]), sequence])),
                                  initial_state=h, training=True)
      h = [h1, h2] # h = final state
    return self.classifier(tf.reshape(tf.stack(h, axis=-1), shape=[32, -1]))

  def call(self, input, training):
    xs = self.encoder(input, return_dict=True).hidden_states

    if training:
      return self._forward(xs)
    else:
      ys = []
      for i in range(self.num_forward_passes):
        ys.append(self._forward(xs))
      return tf.keras.layers.Average()(ys)



# functions that produce distributions for weights in DenseVariational layer of VGRUModel
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model =  tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

"""
  GRU-READ with DenseVariational layer as classifier
"""
class VGRUModel(tf.keras.Model):
  def __init__(self, model_name, num_labels, train_data_size, num_forward_passes=1000, num_gru_units=32):
    super().__init__()

    self.num_labels = num_labels
    self.num_forward_passes = num_forward_passes
    self.num_gru_units = num_gru_units

    self.encoder = TFAutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name, output_hidden_states=True))
    self.encoder.trainable = False

    self.linear = tf.keras.layers.Dense(2 * self.num_gru_units) # joiner network
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_gru_units, return_sequences=True, return_state=True))

    self.norm = tf.keras.layers.LayerNormalization()

    self.dense_var = tfp.layers.DenseVariational(self.num_labels, activation='softmax',
                                                 make_posterior_fn=posterior, make_prior_fn=prior,
                                                 kl_weight=1/train_data_size)

  def call(self, input, training):
    xs = self.encoder(input, return_dict=True).hidden_states
    batchsize = xs[0].shape[0]
    h = None
    sequence = tf.zeros(shape=(xs[0].shape[0], 2 * self.num_gru_units))
    for i in range(len(xs)):
      sequence, h1, h2 = self.gru(self.norm(tf.keras.layers.Add()([self.linear(xs[i]), sequence])), initial_state=h)
      h = [h1, h2]
    x = tf.reshape(tf.stack(h, axis=-1), shape=[batchsize, -1])

    if training:
      return self.dense_var(x)
    else:
      probs_samples = []
      for _ in range(self.num_forward_passes):
        probs_samples.append(self.dense_var(x))
      return tf.keras.layers.Average()(probs_samples)
