import tensorflow as tf
import tensorflow_probability as tfp
import keras_nlp
import transformers
from transformers import AutoTokenizer, TFAutoModel, AutoConfig, TFAutoModelForSequenceClassification

from gru_models import GRUModel, VGRUModel, READ_GRU

def get_model(model_name, learning_rate, use_model="basemodel", tpu_strategy=None, num_gru_units=8, train_data_size=None):
  """ loads the model and compiles it with the passed hyperparams.
  Which model to use is chosen based on use_model.
  returns a model ready to train.
  """
  
  assert use_model == "basemodel" or use_model == "read" or use_model == "read-var" or use_model == "paper-read", "invalid model name, use 'basemodel', 'read', 'read-var' or 'paper-read'"

  # if read-var, assert train_data_size is not None
  if use_model == "read-var":
    assert train_data_size is not None, "train_data_size must be specified for read-var model"

  if tpu_strategy is None:
    if use_model == "basemodel":
      model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name))
      model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
          metrics=[tf.metrics.SparseCategoricalAccuracy()]
      )
    elif use_model == "read":
      model = GRUModel(model_name, 2, num_gru_units=num_gru_units)
      model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
          metrics=[tf.metrics.SparseCategoricalAccuracy()]
      )
    elif use_model == "paper-read":
      model = READ_GRU(model_name=model_name, num_labels=2, hidden_dimension=num_gru_units)
      model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
          metrics=[tf.metrics.SparseCategoricalAccuracy()]
      )  
    else:
      model = VGRUModel(model_name, 2, train_data_size=train_data_size, num_gru_units=num_gru_units)
      model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
          metrics=[tf.metrics.SparseCategoricalAccuracy()]
      )  
      
  else: # tpu included
    if use_model == "basemodel":
      with tpu_strategy.scope():
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name))
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]
        )
    
    elif use_model == "read":
      with tpu_strategy.scope():
        model = GRUModel(model_name, 2, num_gru_units=8)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]
        )

    elif use_model == "paper-read":
      with tpu_strategy.scope():
        model = READ_GRU(model_name=model_name, num_labels=2, hidden_dimension=num_gru_units)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]
        )  
    
    else:
      with tpu_strategy.scope():
        model = VGRUModel(model_name, 2, train_data_size=train_data_size, num_gru_units=8)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=None),
            metrics=[tf.metrics.SparseCategoricalAccuracy()]
        )

  return model
