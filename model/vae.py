import tensorflow as tf
from tensorflow_probability import distributions as tfd


def vae_cost(y_true, y_pred, mu, sigma, z_sample, analytic_kl=True, kl_weight=1):
    # compute cross entropy loss for each dimension of every datapoint
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                        logits=y_pred)  

    # compute cross entropy loss for all instances in mini-batch
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy)
    # compute reverse KL divergence, either analytically 
    # or through MC approximation with one sample
    if analytic_kl:
        kl_divergence = - 0.5 * tf.math.reduce_sum(
            1 + tf.math.log(tf.math.square(sigma)) - tf.math.square(mu) - tf.math.square(sigma),
            axis=1)  
    else:
        logpz = tfd.Normal(loc=0., scale=1.).prob(z_sample)
        logqz_x = tfd.Normal(loc=mu, scale=tf.math.square(sigma)).prob(z_sample) 
        kl_divergence = logqz_x - logpz
    elbo = tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood)  
    return -elbo


class Sampling(tf.keras.layers.Layer):
  def call(sellf, inputs):
    mu, sigma = inputs
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return mu + tf.exp(0.5 * sigma) * epsilon

def encoder_layers(inputs, latent_dim, dims:list, activation:str):
    encoder = tf.keras.layers.Dense(units=dims[0], activation=activation)(inputs)
    encoder = tf.keras.layers.Dropout(0.2)(encoder)

    for dim in dims:
        encoder = tf.keras.layers.Dense(units=dim, activation=activation)(encoder)
        encoder = tf.keras.layers.Dropout(0.2)(encoder)
 
    mu = tf.keras.layers.Dense(units=latent_dim, activation=activation)(encoder)
    sigma = tf.keras.layers.Dense(units=latent_dim, activation=activation)(encoder)
    return mu, sigma, encoder

def encoder_model(input_shape, latent_dim, dims, activation='tanh'):
  input = tf.keras.layers.Input(shape=input_shape, name='encoder_model_input')
  mu, sigma, encoder = encoder_layers(inputs=input, latent_dim=latent_dim, dims=dims, activation=activation)
  z = Sampling()((mu, sigma))
  model = tf.keras.Model(inputs=input, outputs=[mu, sigma, z])
  model._name = 'Encoder'
  return model

def decoder_layers(inputs, dims, activation):
    dec = tf.keras.layers.Dense(units=dims[0], activation=activation)(inputs)
    dec = tf.keras.layers.Dropout(0.2)(dec)
    
    for dim in dims[1:-1]:
        dec = tf.keras.layers.Dense(units=dim, activation=activation)(dec)
        dec = tf.keras.layers.Dropout(0.2)(dec)

    dec = tf.keras.layers.Dense(units=dims[-1], activation=activation)(dec)

    return dec

def decoder_model(input_shape, dims, activation='tanh'):
  inputs = tf.keras.layers.Input(shape=input_shape)
  outputs = decoder_layers(inputs, dims, activation)
  model = tf.keras.Model(inputs, outputs)
  model._name = 'Decoder'
  return model

def vae(encoder, decoder, bert, input_shape):
 input_ids = tf.keras.layers.Input(shape=input_shape, name='input_ids', dtype=tf.int32)
 attention_mask = tf.keras.layers.Input(shape=input_shape, name='attention_mask', dtype=tf.int32)
 token_type_ids = tf.keras.layers.Input(shape=input_shape, name='token_type_ids', dtype=tf.int32)
 embeddings = bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0]

 mu, sigma, z = encoder(embeddings)
 reconstructed = decoder(z)

 model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=reconstructed)

 loss = vae_cost(embeddings, reconstructed, mu, sigma, z)
 model.add_loss(loss)
 model._name = 'VAE'
 return model






