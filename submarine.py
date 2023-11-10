import numpy as np


class Submarine:
    def __init__(self, position=(0, 0), heading=0, max_speed=25, turn_radius=500):
        self.max_speed = max_speed  # in knots
        self.position = np.array(position)  # in yards
        self.current_speed = 0  # in knots
        self.rudder = 0  # between 1 (full right) and -1 (full left)
        self.heading = heading  # in degrees
        self.desired_heading = heading  # in degrees
        self.turn_radius = turn_radius  # in yards

    def set_speed(self, speed):
        if speed > self.max_speed:
            self.current_speed = self.max_speed
        elif speed < 0:
            self.current_speed = 0
        else:
            self.current_speed = speed

    def set_position(self, position):
        self.position = position

    def set_rudder(self, rudder):
        self.rudder = rudder

    def move(self, time):
        turn_rate = self.rudder * self.current_speed * 0.562603 * 180 / (np.pi * self.turn_radius)  # in degrees/sec
        self.heading = self.heading + turn_rate * time
        self.heading = self.heading % 360
        self.position = self.position + self.current_speed * time * 0.562603 \
            * np.array([np.sin(self.heading * np.pi / 180), np.cos(self.heading * np.pi / 180)])
        # calculate rudder
        degree_difference = (self.desired_heading - self.heading + 180) % 360 - 180
        if abs(degree_difference) < (abs(turn_rate) * time):
            self.rudder = degree_difference / time * self.rudder/turn_rate*.9
        else:
            self.rudder = np.sign(degree_difference)
        # time passage
        # time in seconds, convert to yards/sec (multiply by 0.562603) then multiply by time


    def aim_for(self, desired_heading):
        self.desired_heading = desired_heading

    def bearing_to(self, target):
        target_position = target.get_position()
        return (np.arctan2(target_position[0] - self.position[0],
                           target_position[1] - self.position[1]) * 180 / np.pi) % 360

    def distance_to(self, target):
        target_position = target.get_position()
        return np.sqrt((target_position[1] - self.position[1]) ** 2 + (target_position[0] - self.position[0]) ** 2)

    def get_position(self):
        return self.position

    def get_heading(self):
        return self.heading

    def get_desired_heading(self):
        return self.desired_heading

    def get_speed(self):
        return self.current_speed

    def get_rudder(self):
        return self.rudder
