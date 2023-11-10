from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from submarine import Submarine

keras = tf.keras


# input: sub1 speed, sub1 heading, bearing to sub2, bearing rate
# output: sub2 speed, sub2 heading, distance to sub2
# 10 second steps, turn on average once every 500 seconds

# we want: input: sub1 x speed/25, sub1 y speed/25, sub2 x bearing normalized, sub2 y bearing normalized
# we want: input: sub1 speed/25, sub2 relative bearing (-180,180)/180, sub2 bearing rate
# output: distance to sub2/10000

def preprocess_input_slice(slice):
    return np.array([slice[0]/25, ((slice[2]-slice[1]+180) % 360)/180 - 1, slice[3]])


def preprocess_output_slice(slice):
    return slice[2]


def to_windows(input_history, output_history, window_size=30, drop_rate=0.9):
    input_windows = []
    output_windows = []
    for i in range(len(input_history)):
        if np.random.random() > drop_rate:
            current_window = []
            for j in range(i-window_size+1, i+1):
                if j < 0:
                    current_window.append([0, 0, 0, 0])
                else:
                    current_window.append(preprocess_input_slice(input_history[j]))
            input_windows.append(current_window)
            output_windows.append(preprocess_output_slice(output_history[i]))
    return np.array(input_windows), np.array(output_windows)

def to_training_windows(input_history, output_history, window_size=30, drop_rate=0.9):
    input_windows = []
    output_windows = []
    for i in range(len(input_history)):
        if np.random.random() > drop_rate:
            current_input_window = []
            current_output_window = []
            for j in range(i-window_size+1, i+1):
                if j < 0:
                    current_input_window.append([0, 0, 0, 0])
                    current_output_window.append(0)
                else:
                    current_input_window.append(preprocess_input_slice(input_history[j]))
                    current_output_window.append(preprocess_output_slice(output_history[j]))
            input_windows.append(current_input_window)
            output_windows.append(current_output_window)
    return np.array(input_windows), np.array(output_windows)


def to_stateful_windows(input_history, output_history, window_size=30):
    input_windows = []
    output_windows = []
    for i in range(int(len(input_history)/window_size)):
        current_input_window = []
        current_output_window = []
        for j in range(i*window_size, (i+1)*window_size):
            current_input_window.append(preprocess_input_slice(input_history[j]))
            current_output_window.append(preprocess_output_slice(output_history[j]))
        input_windows.append(current_input_window)
        output_windows.append(current_output_window)
    return np.array(input_windows), np.array(output_windows)


def save_windows(input_windows, output_windows, filename):
    np.savez(filename, input_windows=input_windows, output_windows=output_windows)
