import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import exists
from simulation import make_sim_data, save_sim_data
from preprocess import to_windows, save_windows, to_training_windows, to_stateful_windows
keras = tf.keras


def generate_dataset(window_size, num_sims, sim_length, sim_data):
    input_windows = []
    output_windows = []
    num_windows = num_sims * int(sim_length / window_size)
    reporting_interval = max(1, 10 ** (np.floor(np.log10(num_windows)) - 1))
    for i in sim_data:
        current_windows = len(input_windows)
        windows = to_stateful_windows(i[0], i[1], window_size=window_size)
        if windows:  # maybe they all got dropped
            input_windows.extend(windows[0])
            output_windows.extend(windows[1])
        if current_windows // reporting_interval < (len(input_windows)) // reporting_interval:
            print("Window ", len(input_windows), " of ", num_windows)
    input_windows = np.array(input_windows)
    output_windows = np.array(output_windows)
    return input_windows, output_windows


def generate_sim_data(num_sims, sim_length):
    if not exists('sim_data.npy'):
        sim_data = make_sim_data(num_sims, sim_length)
        save_sim_data(sim_data, 'sim_data.npy')
    else:
        sim_data = np.load('sim_data.npy')
        if sim_data.shape != (num_sims, 2, sim_length, 3):
            print("Sim data shape is ", sim_data.shape, " but should be ", (num_sims, 2, sim_length, 3),
                  ". Regenerating.")
            sim_data = make_sim_data(num_sims, sim_length)
            save_sim_data(sim_data, 'sim_data.npy')
    return sim_data


def generate_training_data(window_size, num_sims, sim_length):
    if not exists('training_data.npz'):
        print("Training data not found. Generating.")
        sim_data = generate_sim_data(num_sims, sim_length)
        input_windows, output_windows = generate_dataset(window_size, num_sims, sim_length, sim_data)
        save_windows(input_windows, output_windows, 'training_data.npz')
    else:
        training_data = np.load('training_data.npz')
        input_windows = training_data['input_windows']
        output_windows = training_data['output_windows']
        if input_windows.shape[1] != window_size or output_windows.shape[1] != window_size:
            print("Window size is ", input_windows.shape[2], " but should be ", window_size, ". Regenerating.")
            sim_data = generate_sim_data(num_sims, sim_length)
            input_windows, output_windows = generate_dataset(window_size, num_sims, sim_length, sim_data)
            save_windows(input_windows, output_windows, 'training_data.npz')
        elif input_windows.shape[0] != num_sims * int(sim_length / window_size):
            print("Number of windows is ", input_windows.shape[0], " but should be ",
                  num_sims * int(sim_length / window_size), ". Regenerating.")
            sim_data = generate_sim_data(num_sims, sim_length)
            input_windows, output_windows = generate_dataset(window_size, num_sims, sim_length, sim_data)
            save_windows(input_windows, output_windows, 'training_data.npz')
    x_train = np.array(input_windows[:int(len(input_windows)*0.8)])
    y_train = np.array(output_windows[:int(len(output_windows)*0.8)])
    x_test = np.array(input_windows[int(len(input_windows)*0.8):])
    y_test = np.array(output_windows[int(len(output_windows)*0.8):])
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_dataset, test_dataset
