import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import preprocess_input_slice
from submarine import Submarine
keras = tf.keras

sim_time = 300
step_length = 10

# load the ranger model
ranger = keras.models.load_model('ranger.h5')

ranger.reset_states()
print(ranger.summary())

sub1 = Submarine()
sub2 = Submarine((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)), 180)

sub1.set_speed(np.random.randint(5, 25))
sub2.set_speed(np.random.randint(5, 25))
sub1.aim_for(np.random.randint(0, 360))
sub2.aim_for(np.random.randint(0, 360))
history = []
input_history = []
for i in range(sim_time):
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
    history.append([sub1.get_position(), sub2.get_position()])
    input_history.append([sub1.get_speed(),
                          sub1.get_heading(),
                          sub1.bearing_to(sub2),
                          ((sub1.bearing_to(sub2)-last_bearing+180) % 360-180)/step_length])

processed_input = []
for i in input_history:
    processed_input.append(preprocess_input_slice(i))

processed_input = np.array(processed_input)
processed_input = processed_input.reshape((1, processed_input.shape[0], processed_input.shape[1]))

prediction = ranger.predict(processed_input)[0]
prediction = np.array(prediction)
prediction = prediction.reshape((prediction.shape[0], prediction.shape[1]))

prediction_x = []
prediction_y = []
for i in range(len(prediction)):
    prediction_x.append(history[i][0][0]+prediction[i][0]*np.sin(input_history[i][2]*np.pi/180))
    prediction_y.append(history[i][0][1]+prediction[i][0]*np.cos(input_history[i][2]*np.pi/180))


x1 = []
y1 = []
x2 = []
y2 = []

sae = 0
for index, i in enumerate(history):
    x1.append(i[0][0])
    y1.append(i[0][1])
    x2.append(i[1][0])
    y2.append(i[1][1])
    print("predicted range: ", prediction[min(index, len(prediction)-1)][0], " actual range: ", np.sqrt((i[0][0]-i[1][0])**2+(i[0][1]-i[1][1])**2))
    sae += np.abs(prediction[min(index, len(prediction)-1)][0] - np.sqrt((i[0][0]-i[1][0])**2+(i[0][1]-i[1][1])**2))

print("mean absolute error: ", sae/len(history))

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(x1, y1, 'b')
ax.plot(x2, y2, 'r')
ax.plot(prediction_x, prediction_y, 'g')

plt.show()

print('done')

