# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import time
import math
import numpy as np

class Controller():
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.CTRL = pygame.joystick.Joystick(0)
        self.CTRL.init()
        self.LJ = [0,0]
        self.RJ = [0,0]
        self.LT = 0
        self.RT = 0
        self.a_held = 0
        self.x_held = 0
        self.b_held = 0
        self.y_held = 0

        self.request = np.array([0 , 0, 327, 0, 0, 0])
        self.output = np.array([0.0001 , 0.0001, 327, 0, 0, 0])
        self.maxSpeeds = [1.0, math.pi/64]        # mm/s / rads/s per update
        self.min    = np.array([-200, -200, 0, -math.pi, -math.pi, 0])
        self.max    = np.array([200, 200, 327, math.pi, math.pi, 0])
        self.deadzone = 0.15                   # fraction of stick/trigger range to ignore near rest

    def update(self):
        '''Poll the controller hardware: raw stick/trigger/button state, deadzone-filtered
            but NOT yet speed-scaled - see process() for the actual request math.'''
        pygame.event.pump()
        self.LJ = self._deadzone2(np.array([self.CTRL.get_axis(0), -self.CTRL.get_axis(1)]), self.deadzone) # Position Input (rotated 90 degrees)
        self.RJ = self._deadzone2(np.array([self.CTRL.get_axis(4), -self.CTRL.get_axis(3)]), self.deadzone) # Rotation Input
        self.LT = self._deadzone1((self.CTRL.get_axis(2) + 1)/2, self.deadzone)                              # From -1 to 1
        self.RT = self._deadzone1((self.CTRL.get_axis(5) + 1)/2, self.deadzone)
        self.a_held = self.CTRL.get_button(0)
        self.x_held = self.CTRL.get_button(2)
        self.b_held = self.CTRL.get_button(1)
        self.y_held = self.CTRL.get_button(3)

    def display(self):
        print(f"Left Joystick: {self.LJ[0]:.3f}, {self.LJ[1]:.3f}    Right Joystick: {self.RJ[0]:.3f}, {self.RJ[1]:.3f}\n"+
               f"L/R Trigger: {self.LT:.2f}, {self.RT:.2f}   A: {self.a_held} X: {self.x_held} B: {self.b_held} Y: {self.y_held}")

    def process(self) -> np.ndarray:
        '''Turn the raw, deadzone-filtered input from update() into a
            speed-scaled, accumulated, clamped 6-DOF request.'''
        UpPower = self.maxSpeeds[0] * (self.RT - self.LT)
        self.request = self.request + np.array([
            -self.maxSpeeds[0]*self.LJ[1], self.maxSpeeds[0]*self.LJ[0],
            UpPower,
            self.maxSpeeds[1]*self.RJ[0], self.maxSpeeds[1]*self.RJ[1],
            0
        ])
        self.output = self.clamp()
        #print(self.output)
        return self.output

    def clamp(self):
        temp = [self.round_sig(x) for x in self.request]
        return np.clip(temp, self.min, self.max)

    @staticmethod
    def round_sig(x, sig=3):
        if x == 0:
            return 0.0
        return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))

    @staticmethod
    def _deadzone2(vec: np.ndarray, threshold: float) -> np.ndarray:
        ''' Radial deadzone for a 2D stick: zero out below `threshold`
            magnitude, rescale above it so the output still reaches the
            full original range just past the deadzone edge instead of
            jumping there.'''
        mag = np.linalg.norm(vec)
        if mag < threshold:
            return np.zeros(2)
        return vec * (mag - threshold) / (mag * (1 - threshold))

    @staticmethod
    def _deadzone1(value: float, threshold: float) -> float:
        ''' Same idea as _deadzone2, for a single-axis input (the triggers). '''
        if abs(value) < threshold:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - threshold) / (1 - threshold)

if __name__ == "__main__":
    controller = Controller()
    while True:
        controller.update()
        controller.process()
        #controller.display()
        time.sleep(0.05)
    
    
