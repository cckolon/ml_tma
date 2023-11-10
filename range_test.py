import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from generate_training_data import generate_training_data

keras = tf.keras

sim_time = 300

# load the ranger model
ranger = keras.models.load_model('ranger_training.h5')

_, test_dataset = generate_training_data(150, 1000, 301)
print(ranger.summary())
test_dataset = test_dataset.batch(1)

# get the first test dataset
test_input, test_output = next(iter(test_dataset.shuffle(buffer_size=1000)))

# test the model
output_array = test_output.numpy()
test_output_pred = ranger.predict(test_input)

mean_absolute_error = np.mean(np.abs(output_array - test_output_pred))
print(f'Mean absolute error: {mean_absolute_error}')

# plot the results
plt.plot(output_array[0], label='actual')
plt.plot(test_output_pred[0], label='predicted')
plt.legend()
plt.show()
