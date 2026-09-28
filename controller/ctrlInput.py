'''Xbox controller input for trajectory 3: the sticks and triggers move a 6-DOF tool-pose request
(left stick X/Y, triggers Z, right stick Rx/Ry), Y ends the task.
'''
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, so this file can be run directly
# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import time
import math
import numpy as np
from Core.settings import loadRobotGeometry

GEOMETRY = loadRobotGeometry()
HOME_Z = 2*GEOMETRY["sectL"] + GEOMETRY["toolLength"]      # straight tool-tip height (327 mm)

class Controller():
    def __init__(self):
        '''Opens joystick 0 and starts the request at the straight pose.'''
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.leftStick = [0,0]
        self.rightStick = [0,0]
        self.leftTrigger = 0
        self.rightTrigger = 0
        self.aHeld = 0
        self.xHeld = 0
        self.bHeld = 0
        self.yHeld = 0

        self.request = np.array([0 , 0, HOME_Z, 0, 0, 0])
        self.output = np.array([0.0001 , 0.0001, HOME_Z, 0, 0, 0])
        self.maxSpeeds = [1.0, math.pi/64]        # mm/s / rads/s per update
        self.min    = np.array([-200, -200, 0, -math.pi, -math.pi, 0])
        self.max    = np.array([200, 200, HOME_Z, math.pi, math.pi, 0])
        self.deadzone = 0.15                   # fraction of stick/trigger range to ignore near rest

    def update(self):
        '''Poll the controller hardware: raw stick/trigger/button state, deadzone-filtered
            but NOT yet speed-scaled - see calcTargGlob() for the actual request math.'''
        pygame.event.pump()
        self.leftStick = self._applyDeadzone2D(np.array([self.joystick.get_axis(0), -self.joystick.get_axis(1)]), self.deadzone) # Position Input (rotated 90 degrees)
        self.rightStick = self._applyDeadzone2D(np.array([self.joystick.get_axis(4), -self.joystick.get_axis(3)]), self.deadzone) # Rotation Input
        self.leftTrigger = self._applyDeadzone1D((self.joystick.get_axis(2) + 1)/2, self.deadzone)                              # From -1 to 1
        self.rightTrigger = self._applyDeadzone1D((self.joystick.get_axis(5) + 1)/2, self.deadzone)
        self.aHeld = self.joystick.get_button(0)
        self.xHeld = self.joystick.get_button(2)
        self.bHeld = self.joystick.get_button(1)
        self.yHeld = self.joystick.get_button(3)

    def printInputs(self):
        '''Prints the current stick, trigger and button state.'''
        print(f"Left Joystick: {self.leftStick[0]:.3f}, {self.leftStick[1]:.3f}    Right Joystick: {self.rightStick[0]:.3f}, {self.rightStick[1]:.3f}\n"+
               f"L/R Trigger: {self.leftTrigger:.2f}, {self.rightTrigger:.2f}   A: {self.aHeld} X: {self.xHeld} B: {self.bHeld} Y: {self.yHeld}")

    def calcTargGlob(self) -> np.ndarray:
        '''Turns the raw, deadzone-filtered input from update() into a
            speed-scaled, accumulated, clamped 6-DOF request [x, y, z, Rx, Ry, Rz].'''
        upPower = self.maxSpeeds[0] * (self.rightTrigger - self.leftTrigger)
        self.request = self.request + np.array([
            -self.maxSpeeds[0]*self.leftStick[1], self.maxSpeeds[0]*self.leftStick[0],
            upPower,
            self.maxSpeeds[1]*self.rightStick[0], self.maxSpeeds[1]*self.rightStick[1],
            0
        ])
        self.output = self.clampRequest()
        #print(self.output)
        return self.output

    def clampRequest(self):
        '''The request rounded to 3 significant figures and clamped to [min, max].'''
        temp = [self.roundSignificant(x) for x in self.request]
        return np.clip(temp, self.min, self.max)

    @staticmethod
    def roundSignificant(_x, _sig=3):
        '''_x rounded to _sig significant figures.'''
        if _x == 0:
            return 0.0
        return round(_x, -int(math.floor(math.log10(abs(_x)))) + (_sig - 1))

    @staticmethod
    def _applyDeadzone2D(_vec: np.ndarray, _threshold: float) -> np.ndarray:
        '''Radial deadzone for a 2D stick: zero out below `threshold`
            magnitude, rescale above it so the output still reaches the
            full original range just past the deadzone edge instead of
            jumping there.'''
        mag = np.linalg.norm(_vec)
        if mag < _threshold:
            return np.zeros(2)
        return _vec * (mag - _threshold) / (mag * (1 - _threshold))

    @staticmethod
    def _applyDeadzone1D(_value: float, _threshold: float) -> float:
        '''Same idea as _applyDeadzone2D, for a single-axis input (the triggers).'''
        if abs(_value) < _threshold:
            return 0.0
        sign = 1.0 if _value > 0 else -1.0
        return sign * (abs(_value) - _threshold) / (1 - _threshold)

if __name__ == "__main__":
    controller = Controller()
    while True:
        controller.update()
        controller.calcTargGlob()
        #controller.display()
        time.sleep(0.05)
    
    
