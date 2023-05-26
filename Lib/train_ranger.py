import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from generate_training_data import generate_training_data
keras = tf.keras

window_size = 150
num_sims = 10001
sim_length = 301
adjust_lr = False
load_model = True
epochs = 100
lr = 1e-5
dropout_rate = .1

train_dataset, test_dataset = generate_training_data(window_size, num_sims, sim_length)

if load_model:
    model = keras.models.load_model('ranger_training.h5', compile=False)
else:
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=[None, 3], return_sequences=True, stateful=False),
        keras.layers.LSTM(64, return_sequences=True, stateful=False),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1),
        keras.layers.Lambda(lambda x: 10000*x)
        ])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=keras.losses.MeanSquaredError(),
              metrics=['mean_absolute_error', 'mean_squared_error'])

print(model.summary())

def train(batch_size, epochs=epochs):
    train_batches = train_dataset.batch(batch_size).cache().prefetch(1)
    test_batches = test_dataset.batch(batch_size).cache().prefetch(1)
    reset_states = keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: model.reset_states())
    model_checkpoint = keras.callbacks.ModelCheckpoint('ranger_training.h5', monitor='val_loss', save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_batches, epochs=epochs, validation_data=test_batches, batch_size=batch_size,
                        callbacks=[reset_states, model_checkpoint, early_stopping], shuffle=False)
    return history


if adjust_lr:
    min_lr = 1e-6
    max_lr = 1e-2
    train_batches = train_dataset.batch(32).cache().prefetch(1)
    test_batches = test_dataset.batch(32).cache().prefetch(1)
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: min_lr * 10 ** (epoch / epochs * np.log10(max_lr / min_lr)))
    history = model.fit(train_batches, epochs=epochs,
                        validation_data=test_batches, callbacks=[lr_schedule], batch_size=1)
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([min_lr, max_lr, 0, history.history["loss"][epochs//10]])
    plt.savefig('lr.png')
else:
    loss_history = []
    val_loss_history = []
    for batch_size in [256, 128, 64, 32]:
        print(f'Batch size: {batch_size}')
        history = train(batch_size)
        model = keras.models.load_model('ranger_training.h5')
        loss_history.extend(history.history['loss'])
        val_loss_history.extend(history.history['val_loss'])
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    model.save('ranger.h5')
plt.show()


