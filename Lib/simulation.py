import numpy as np
from submarine import Submarine
import matplotlib.pyplot as plt


# input: sub1 speed, sub1 heading, bearing to sub2, bearing rate to sub2
# output: sub2 speed, sub2 heading, distance to sub2
# 10 second steps, turn on average once every 500 seconds
def run_simulation(steps, step_length=10):
    sub1 = Submarine()
    sub2 = Submarine((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)), 180)
    sub1.set_speed(10)
    sub2.set_speed(10)
    sub1.aim_for(np.random.randint(0, 360))
    sub2.aim_for(np.random.randint(0, 360))
    input_history = []
    output_history = []
    for i in range(steps):
        last_bearing = sub1.bearing_to(sub2)
        sub1.move(step_length)
        sub2.move(step_length)
        choice = np.random.randint(0, 5)
        if choice == 0:
            sub1.aim_for(sub1.get_heading())
        elif choice == 1:
            sub1.aim_for(sub1.get_heading() - 10)
        elif choice == 2:
            sub1.aim_for(sub1.get_heading() + 10)
        elif choice == 3:
            sub1.set_speed(sub1.get_speed() + 1)
        elif choice == 4:
            sub1.set_speed(sub1.get_speed() - 1)
        if np.random.randint(0, 100) == 0:
            sub2.aim_for(np.random.randint(0, 360))
        input_history.append([sub1.get_speed(),
                              sub1.get_heading(),
                              sub1.bearing_to(sub2),
                              ((sub1.bearing_to(sub2)-last_bearing+180) % 360-180)/step_length])
        output_history.append([sub2.get_speed(),
                               sub2.get_heading(),
                               sub1.distance_to(sub2),
                               0])
    return input_history, output_history


def make_sim_data(num_sims, sim_length):
    history = []
    reporting_interval = max(1,10**(np.floor(np.log10(num_sims))-1))
    for i in range(num_sims):
        if i % reporting_interval == 0:
            print("Simulation ", i, " of ", num_sims)
        input_history, output_history = run_simulation(sim_length)
        history.append([input_history, output_history])
    return np.array(history)


def save_sim_data(history, filename):
    np.save(filename, history)

