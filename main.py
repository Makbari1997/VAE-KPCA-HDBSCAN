import os
import json
from pathlib import Path

from data_modules.data_utils import *
from data_modules.DataLoader import DataLoader

from model.model_utils import *
from model.encoder import *
from model.vae import *
from model.train import *

from utils import *

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run(config):
    path_bert = Path('./artifacts/{}/bert/'.format(config['dataset']))
    path_vae = Path('./artifacts/{}/vae/'.format(config['dataset']))
    path_bert.mkdir(parents=True, exist_ok=True)
    path_vae.mkdir(parents=True, exist_ok=True)

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
        path=os.path.join('dataset', config['dataset'], 'in_lbl_2_indx.txt'), 
        intents=train_intents + test_intents + dev_intents
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

    train_tf = to_tf_format((train_input_ids, train_attention_mask, train_token_type_ids), None, len(train_sentences), batch_size=16)
    test_tf = to_tf_format((test_input_ids, test_attention_mask, test_token_type_ids), None, len(test_sentences), batch_size=1)
    dev_tf = to_tf_format((dev_input_ids, dev_attention_mask, dev_token_type_ids), None, len(dev_sentences), batch_size=1)
    ood_tf = to_tf_format((ood_input_ids, ood_attention_mask, ood_token_type_ids), None, len(ood_sentences), batch_size=1)
    print('Data preparation finished successfully!')

    print('------------------------------------------------------------------')

    print('Finetuning of bert is in progress...')
    classifier = finetune(
        x_train=train_sentences + dev_sentences, y_train=np.concatenate((train_intents_encoded, dev_intents_encoded), axis=0),
        x_validation=test_sentences, y_validation=test_intents_encoded,
        max_length=max_length, num_labels=len(np.unique(np.array(train_intents))), path=os.path.join('artifacts', config['dataset'], 'bert/'), 
        train=config['finetune'], first_layers_to_freeze=11, num_epochs=config['finetune_epochs'], model_name=config['bert']
    )
    classifier.load_weights(os.path.join('artifacts', config['dataset'], 'bert/'))
    bert.layers[0].set_weights(classifier.layers[0].get_weights())
    print('Finetuning finished successfully and weights saved to {}'.format(os.path.join('artifacts', config['dataset'], 'bert/')))

    print('------------------------------------------------------------------')

    print('VAE model creation is in progress...')
    model = vae(
        bert=bert,
        encoder=encoder_model((config['vector_dim'],), config['latent_dim'], dims=config['encoder'], activation=config['activation']),
        decoder=decoder_model((config['latent_dim'],), dims=config['decoder'], activation=config['activation']),
        input_shape=((max_length,))
    )

    model.layers[3].trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['vae_learning_rate'])
    train_loss_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()
    print('Model is created successfully!')

    print('------------------------------------------------------------------')

    print('Training of VAE is in progress...')
    train_loop(
        model, 
        optimizer, 
        train_tf, 
        dev_tf, 
        path=os.path.join('artifacts', config['dataset'], 'vae', 'vae.h5'), 
        batch_size=config['batch_size'], 
        num_epochs=config['train_epochs'], 
        train_loss_metric=train_loss_metric, val_loss_metric=val_loss_metric
    )
    model.load_weights(os.path.join('artifacts', config['dataset'], 'vae', 'vae.h5'))
    print('Training is done and weights saved to {}'.format(os.path.join('artifacts', config['dataset'], 'vae', 'vae.h5')))

    print('------------------------------------------------------------------')

    print('Calculating train and dev loss for visualization...')
    train_tf = to_tf_format((train_input_ids, train_attention_mask, train_token_type_ids), None, len(train_sentences), batch_size=1)
    train_loss = compute_loss(model, train_tf)
    dev_loss = compute_loss(model, dev_tf)
    train_loss_normalized = normalize(train_loss, path=os.path.join('artifacts', config['dataset']), mode='train') 
    dev_loss_normalized = normalize(dev_loss, path=os.path.join('artifacts', config['dataset']), mode='eval')
    visualize(train_loss_normalized, os.path.join('artifacts', config['dataset'], 'vae_loss_for_{}_train.png'.format(config['dataset'])))
    visualize(dev_loss_normalized, os.path.join('artifacts', config['dataset'], 'vae_loss_for_{}_dev.png'.format(config['dataset'])))
    print('You can use figures in {} to decide what threshold should be used.'.format(os.path.join('artifacts', config['dataset'])))

    print('------------------------------------------------------------------')



if __name__ == '__main__':
    config_file = open('./config.json')
    config = json.load(config_file)

    run(config)