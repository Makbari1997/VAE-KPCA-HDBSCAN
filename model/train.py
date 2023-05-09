import time
import numpy as np
import tensorflow as tf

@tf.function
def train_step(model, optimizer, x, y, z):
  with tf.GradientTape() as tape:
    logits = model([x, y, z], training=True)
    loss_value = model.losses
  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss_value

@tf.function
def test_step(model, x, y, z):
  logits = model([x, y, z], training=False)
  loss_value = model.losses
  return loss_value

def train_loop(model, optimizer, train_data, val_data, path, batch_size, num_epochs, train_loss_metric, val_loss_metric, lr_decrease_step=50):
  best_val_loss = np.inf
  for epoch in range(num_epochs):
    print("\nStart of epoch %d" % (epoch,))
    
    start_time = time.time()
    # Iterate over the batches of the dataset.
    for step, (x, y, z) in enumerate(train_data):
        loss_value = train_step(model, optimizer, x, y, z)
        train_loss_metric.update_state(loss_value)
        if step % 50 == 0:
          print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step, float(train_loss_metric.result().numpy()))
          )
          print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    print("Training loss over epoch: %.4f" % (float(train_loss_metric.result().numpy()),))
    train_loss_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x, y, z in val_data:
        val_loss_metric.update_state(test_step(model, x, y, z))

    print("Validation loss: %.4f" % (float(val_loss_metric.result().numpy()),))

    if best_val_loss > val_loss_metric.result().numpy():
      print('Validation loss has improved form {} to {}\nSaving weights...'.format(best_val_loss, val_loss_metric.result().numpy()))
      best_val_loss = val_loss_metric.result().numpy()
      model.save_weights(filepath=path, overwrite=True, save_format='h5')
    else:
      print('Validation loss has not improved from {}'.format(best_val_loss))
    val_loss_metric.reset_states()

    print("Time taken: %.2fs" % (time.time() - start_time))

def compute_loss(model, data):
  loss = []
  for step, (x, y, z) in enumerate(data):
    loss.append(test_step(model, x, y, z)[0].numpy())
  return loss