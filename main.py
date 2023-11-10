import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from preprocess import preprocess_input_slice
from submarine import Submarine

keras = tf.keras

num_inputs = 3
num_actions = 5  # do nothing, turn left, turn right, speed up, slow down
num_hidden = 32
plot = True

inputs = keras.layers.Input(shape=(None, num_inputs))
recurrent = keras.layers.SimpleRNN(num_hidden, return_sequences=True, return_state=True)(inputs)
common = keras.layers.Dense(num_hidden, activation='relu')(recurrent[-1])
action = keras.layers.Dense(num_actions, activation='softmax')(common)
critic = keras.layers.Dense(1)(common)

ranger = keras.models.load_model('ranger.h5')
decider = keras.models.Model(inputs=[inputs], outputs=[action, critic])

optimizer = keras.optimizers.Adam(lr=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
steps_per_episode = 300
step_length = 10

sub1 = Submarine()
sub2 = Submarine((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)), 180)
sub2.set_speed(np.random.randint(5, 25))


while True:
    episode_reward = 0
    input_history = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    prediction_x = []
    prediction_y = []
    sub1.set_speed(np.random.randint(5, 25))
    sub2.set_speed(np.random.randint(5, 25))
    sub1.set_position((0, 0))
    sub2.set_position((np.random.randint(-1000, 1000), np.random.randint(-1000, 1000)))
    sub2.aim_for(np.random.randint(0, 360))
    range_table = []
    with tf.GradientTape() as tape:
        for step in range(steps_per_episode):
            last_bearing = sub1.bearing_to(sub2)
            sub1.move(step_length)
            sub2.move(step_length)
            state = np.array([sub1.get_speed(),
                              sub1.get_heading(),
                              sub1.bearing_to(sub2),
                              ((sub1.bearing_to(sub2)-last_bearing+180) % 360-180)/step_length])
            state = preprocess_input_slice(state)
            state = np.expand_dims(state, axis=0)
            input_history.append(state)
            action_probs, critic_value = decider(np.array(input_history))
            critic_value_history.append(critic_value[0, 0])
            action = np.random.choice(num_actions, p=np.squeeze(action_probs[-1]))
            action_probs_history.append(tf.math.log(action_probs[-1, action]))
            if action == 0:
                sub1.aim_for(sub1.get_heading())
            elif action == 1:
                sub1.aim_for(sub1.get_heading()-10)
            elif action == 2:
                sub1.aim_for(sub1.get_heading()+10)
            elif action == 3:
                sub1.set_speed(sub1.get_speed()+1)
                pass
            elif action == 4:
                sub1.set_speed(sub1.get_speed()-1)
                pass
            if np.random.randint(0, 100) == 0:
                sub2.aim_for(np.random.randint(0, 360))
            if plot:
                x1.append(sub1.get_position()[0])
                y1.append(sub1.get_position()[1])
                x2.append(sub2.get_position()[0])
                y2.append(sub2.get_position()[1])
            range_table.append(sub1.distance_to(sub2))
        range_estimate = ranger(np.array(input_history)).numpy().reshape(-1)
        for i in range(len(range_estimate)):
            prediction_x.append(x1[i]+range_estimate[i]*(x2[i]-x1[i])/(((x1[i]-x2[i])**2+(y1[i]-y2[i])**2)**0.5))
            prediction_y.append(y1[i]+range_estimate[i]*(y2[i]-y1[i])/(((x1[i]-x2[i])**2+(y1[i]-y2[i])**2)**0.5))
        range_error = np.array(range_table) - range_estimate
        rewards_history = [1-np.abs(x)/10000 for x in range_error]
        episode_reward = sum(rewards_history)/steps_per_episode
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + 0.99 * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-7)
        returns = returns.tolist()

        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, decider.trainable_variables)
        optimizer.apply_gradients(zip(grads, decider.trainable_variables))

        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
    if plot & (episode_count % 10 == 0):
        plt.plot(x1, y1, 'b')
        plt.plot(x2, y2, 'r')
        plt.plot(prediction_x, prediction_y, 'g')
        plt.savefig('images/{}.png'.format(episode_count))
        plt.clf()

    episode_count += 1
    if episode_count % 2 == 0:
        decider.save('decider.h5')
        print('Episode {} Reward: {}'.format(episode_count, episode_reward))

    if running_reward > 1000:
        print('Solved at episode {}!'.format(episode_count))
        break

