import tensorflow as tf
from model.metrics import f1_m
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer



def __finetune_preprocess__(x, y, model_name, batch_size, max_length):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  x_tokenized = tokenizer(x, return_tensors='tf', padding='max_length', max_length=max_length, truncation=True)
  x_tokenized = {i: x_tokenized[i] for i in tokenizer.model_input_names}
  del tokenizer
  return tf.data.Dataset.from_tensor_slices((x_tokenized, y)).batch(batch_size)

def finetune(x_train, y_train, x_validation, y_validation, max_length, num_labels, path, model_name='bert-base-uncased', lr=2e-5, num_epochs=60, batch_size=16, first_layers_to_freeze=10, train=True):
  if train:
    classifier = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    train_data = __finetune_preprocess__(x_train, y_train, model_name, batch_size, max_length)
    dev_data = __finetune_preprocess__(x_validation, y_validation, model_name, batch_size, max_length)
    for i in range(first_layers_to_freeze):
      classifier.bert.encoder.layer[i].trainable = False
    classifier.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=[f1_m, tf.keras.metrics.CategoricalAccuracy()]
    ) 
    classifier.fit(
    train_data, validation_data=dev_data, epochs=num_epochs,
    callbacks=[
              tf.keras.callbacks.ModelCheckpoint(filepath=path, 
                                                monitor='val_f1_m', 
                                                mode='max', 
                                                save_weights_only=True, 
                                                save_best_only=True)
    ])
    del classifier
  classifier = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
  classifier.load_weights(path)
  return classifier