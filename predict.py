import os
import json

from data_modules.DataLoader import DataLoader
from data_modules.data_utils import *

from model.train import compute_loss
from model.vae import *
from model.model_utils import *
from model.encoder import *

from utils import *

from sklearn import metrics

import numpy as np


def __predict_preprocess__(x, tokenizer, max_length):
  x_tokenized = tokenizer(x, return_tensors='tf', padding='max_length', max_length=max_length, truncation=True)
  return {i: x_tokenized[i] for i in tokenizer.model_input_names}

def predict(classifier:object, tokenizer:object, losses:list, sentences:list, threshold:float, ood_label:int, max_length:int) -> list:
  labels = []
  for loss, sen in zip(losses, sentences):
    if loss <= threshold:
      labels.append(np.argmax(classifier.predict(
          __predict_preprocess__(sen, tokenizer, max_length)
      )[0], axis=1)[0])
    else:
      labels.append(ood_label)
  return labels

def run(config):
    print('Loading data from {}...'.format(os.path.join('data', config['dataset'])))
    dataloader = DataLoader(path=os.path.join('dataset', config['dataset']))
    train_sentences, train_intents = dataloader.train_loader()
    dev_sentences, dev_intents = dataloader.dev_loader()
    test_sentences, test_intents = dataloader.test_loader()
    ood_sentences, ood_intents = dataloader.ood_loader()
    print('Data is loaded successfully!')

    print('------------------------------------------------------------------')

    print('Encoding intent labels...')
    in_lbl_2_indx = get_lbl_2_indx(
        path=os.path.join('dataset', config['dataset'], 'in_lbl_2_indx.txt')
    )

    train_intents_encoded = one_hot_encoder(train_intents, in_lbl_2_indx)
    test_intents_encoded = one_hot_encoder(test_intents, in_lbl_2_indx)
    dev_intents_encoded = one_hot_encoder(dev_intents, in_lbl_2_indx)

    ood_lbl_2_indx = get_lbl_2_indx(
        path=os.path.join('dataset', config['dataset'], 'ood_lbl_2_indx.txt'),
        intents=ood_intents
    )
    ood_intents_encoded  = one_hot_encoder(ood_intents, ood_lbl_2_indx)
    print('Encoding done successfully!')

    print('------------------------------------------------------------------')

    max_length = max_sentence_length(train_sentences, policy=config['seq_length'])

    print('Downloading {}'.format(config['bert']))
    bert, tokenizer = get_bert(config['bert'])
    print('Download finished successfully!')

    print('------------------------------------------------------------------')

    print('Preparing data for bert, it may take a few minutes...')
    train_input_ids, train_attention_mask, train_token_type_ids = preprocessing(tokenizer, train_sentences, max_length)
    test_input_ids, test_attention_mask, test_token_type_ids = preprocessing(tokenizer, test_sentences, max_length)
    dev_input_ids, dev_attention_mask, dev_token_type_ids = preprocessing(tokenizer, dev_sentences, max_length)
    ood_input_ids, ood_attention_mask, ood_token_type_ids = preprocessing(tokenizer, ood_sentences, max_length)

    # train_tf = to_tf_format((train_input_ids, train_attention_mask, train_token_type_ids), None, len(train_sentences), batch_size=1)
    test_tf = to_tf_format((test_input_ids, test_attention_mask, test_token_type_ids), None, len(test_sentences), batch_size=1)
    # dev_tf = to_tf_format((dev_input_ids, dev_attention_mask, dev_token_type_ids), None, len(dev_sentences), batch_size=1)
    ood_tf = to_tf_format((ood_input_ids, ood_attention_mask, ood_token_type_ids), None, len(ood_sentences), batch_size=1)
    print('Data preparation finished successfully!')

    print('------------------------------------------------------------------')

    print('Loading bert weights from {}'.format(os.path.join('artifacts', config['dataset'], 'bert/')))
    classifier = finetune(
        x_train=train_sentences + dev_sentences, y_train=np.concatenate((train_intents_encoded, dev_intents_encoded), axis=0),
        x_validation=test_sentences, y_validation=test_intents_encoded,
        max_length=max_length, num_labels=len(np.unique(np.array(train_intents))), path=os.path.join('artifacts', config['dataset'], 'bert/'), 
        train=config['finetune'], first_layers_to_freeze=11, num_epochs=config['finetune_epochs'], model_name=config['bert']
    )
    classifier.load_weights(os.path.join('artifacts', config['dataset'], 'bert/'))
    bert.layers[0].set_weights(classifier.layers[0].get_weights())
    print('------------------------------------------------------------------')


    print('VAE model creation is in progress...')
    model = vae(
        bert=bert,
        encoder=encoder_model((config['vector_dim'],), config['latent_dim'], dims=config['encoder'], activation=config['activation']),
        decoder=decoder_model((config['latent_dim'],), dims=config['decoder'], activation=config['activation']),
        input_shape=((max_length,))
    )

    model.layers[3].trainable = False
    # optimizer = tf.keras.optimizers.Adam(learning_rate=config['vae_learning_rate'])
    # train_loss_metric = tf.keras.metrics.Mean()
    # val_loss_metric = tf.keras.metrics.Mean()

    model.load_weights(os.path.join('artifacts', config['dataset'], 'vae', 'vae.h5'))
    print('Model was created successfully and weights were loaded from {}.'.format(
        os.path.join('artifacts', config['dataset'], 'vae', 'vae.h5')
    ))

    print('------------------------------------------------------------------')

    test_loss = compute_loss(model, test_tf)
    ood_loss = compute_loss(model, ood_tf)

    eval_loss = test_loss + ood_loss
    eval_sentences = test_sentences + ood_sentences

    normalized_eval_loss = normalize(eval_loss, path=os.path.join('artifacts', config['dataset']), mode='eval')

    visualize(normalized_eval_loss[:len(test_sentences)], os.path.join('artifacts', config['dataset'], 'vae_loss_for_{}_test.png'.format(config['dataset'])))
    visualize(normalized_eval_loss[len(test_sentences):], os.path.join('artifacts', config['dataset'], 'vae_loss_for_{}_ood.png'.format(config['dataset'])))

    y_pred_multiclass = predict(classifier, tokenizer, normalized_eval_loss, eval_sentences, config['ood_threshold'], len(in_lbl_2_indx), max_length)
    y_true_multiclass = [in_lbl_2_indx[i] for i in test_intents] + [len(in_lbl_2_indx)] * len(ood_intents)

    y_true_binary = [0 for i in range(len(test_sentences))] + [1 for i in range(len(ood_sentences))]
    y_pred_binary = [0 if i <= config['ood_threshold'] else 1 for i in normalized_eval_loss]

    print('threshold : {}'.format(config['ood_threshold']))
    print('----------------------------------')
    print('multi class macro f1 : {}'.format(np.round(metrics.f1_score(y_true_multiclass, y_pred_multiclass, average='macro')), 2))
    print('multi class micro f1 : {}'.format(np.round(metrics.f1_score(y_true_multiclass, y_pred_multiclass, average='micro')), 2))
    print('\n')
    print('binary class macro f1 : {}'.format(np.round(metrics.f1_score(y_true_binary, y_pred_binary, average='macro')), 2))
    print('binary class micro f1 : {}'.format(np.round(metrics.f1_score(y_true_binary, y_pred_binary, average='micro')), 2))
    print('------------------------------------------------------------------')


if __name__ == '__main__':
    config_file = open('./config.json')
    config = json.load(config_file)

    run(config)