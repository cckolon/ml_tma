import numpy as np
import matplotlib.pyplot as plt
from submarine import Submarine

sub1 = Submarine()
sub2 = Submarine((np.random.randint(0, 1000), np.random.randint(0, 1000)), 180)

sub1.set_speed(10)
sub2.set_speed(10)
sub1.aim_for(np.random.randint(0, 360))
sub2.aim_for(np.random.randint(0, 360))

bearings = []
ranges = []
x1 = []
y1 = []
x2 = []
y2 = []
for i in range(1000):
    sub1.move(10)
    sub2.move(10)
    if np.random.randint(0, 50) == 0:
        sub1.aim_for(np.random.randint(0, 360))
        sub2.aim_for(np.random.randint(0, 360))
    bearings.append(sub1.bearing_to(sub2))
    ranges.append(sub1.distance_to(sub2))
    x1.append(sub1.get_position()[0])
    y1.append(sub1.get_position()[1])
    x2.append(sub2.get_position()[0])
    y2.append(sub2.get_position()[1])
    print("positions (yards): ", sub1.get_position(), sub2.get_position())
    print(sub1.bearing_to(sub2), sub2.bearing_to(sub1))
    print(sub1.distance_to(sub2), sub2.distance_to(sub1))
    print()

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(x1, y1, 'r')
ax.plot(x2, y2, 'b')
for i in range(0, 1000, 100):
    ax.plot([x1[i], x1[i]+ranges[i]*np.cos(bearings[i]*np.pi/180)], [y1[i], y1[i]+ranges[i]*np.sin(bearings[i]*np.pi/180)], 'g')

plt.show()
