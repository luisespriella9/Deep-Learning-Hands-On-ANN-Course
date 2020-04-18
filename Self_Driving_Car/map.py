# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor_forward_x = NumericProperty(0)
    sensor_forward_y = NumericProperty(0)
    sensor_forward = ReferenceListProperty(sensor_forward_x, sensor_forward_y)
    sensor_left_x = NumericProperty(0)
    sensor_left_y = NumericProperty(0)
    sensor_left = ReferenceListProperty(sensor_left_x, sensor_left_y)
    sensor_right_x = NumericProperty(0)
    sensor_right_y = NumericProperty(0)
    sensor_right = ReferenceListProperty(sensor_right_x, sensor_right_y)
    signal_density_forward = NumericProperty(0)
    signal_density_left = NumericProperty(0)
    signal_density_right = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor_forward = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor_left = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor_right = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal_density_forward = int(np.sum(sand[int(self.sensor_forward_x)-10:int(self.sensor_forward_x)+10, int(self.sensor_forward_y)-10:int(self.sensor_forward_y)+10]))/400.
        self.signal_density_left = int(np.sum(sand[int(self.sensor_left_x)-10:int(self.sensor_left_x)+10, int(self.sensor_left_y)-10:int(self.sensor_left_y)+10]))/400.
        self.signal_density_right = int(np.sum(sand[int(self.sensor_right_x)-10:int(self.sensor_right_x)+10, int(self.sensor_right_y)-10:int(self.sensor_right_y)+10]))/400.
        if self.sensor_forward_x>longueur-10 or self.sensor_forward_x<10 or self.sensor_forward_y>largeur-10 or self.sensor_forward_y<10:
            self.signal_density_forward = 1.
        if self.sensor_left_x>longueur-10 or self.sensor_left_x<10 or self.sensor_left_y>largeur-10 or self.sensor_left_y<10:
            self.signal_density_left = 1.
        if self.sensor_right_x>longueur-10 or self.sensor_right_x<10 or self.sensor_right_y>largeur-10 or self.sensor_right_y<10:
            self.signal_density_right = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal_density_forward, self.car.signal_density_left, self.car.signal_density_right, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor_forward
        self.ball2.pos = self.car.sensor_left
        self.ball3.pos = self.car.sensor_right

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -3
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.3
            if distance < last_distance:
                last_reward = 0.1
        
        if self.car.x == goal_x and self.car.y == goal_y:
            last_reward = 3
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
