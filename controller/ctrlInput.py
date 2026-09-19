import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import time
import math
import PCC_control

class Controller():
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.CTRL = pygame.joystick.Joystick(0)
        self.CTRL.init()

        self.left_thumb = [0,0]
        self.right_thumb = [0,0]
        self.left_trig = 0
        self.right_trig = 0
        self.a_held = 0
        self.x_held = 0
        self.b_held = 0
        self.y_held = 0

    def update(self):
        pygame.event.pump()
        self.left_thumb = [self.CTRL.get_axis(0), -self.CTRL.get_axis(1)]
        self.right_thumb = [self.CTRL.get_axis(3), -self.CTRL.get_axis(4)]
        self.left_trig = self.CTRL.get_axis(2)                              # From -1 to 1
        self.right_trig = self.CTRL.get_axis(5)
        self.a_held = self.CTRL.get_button(0)
        self.x_held = self.CTRL.get_button(2)
        self.b_held = self.CTRL.get_button(1)
        self.y_held = self.CTRL.get_button(3)

    def display(self):
        print(f"Left Joystick: {self.left_thumb[0]:.3f}, {self.left_thumb[1]:.3f}    Right Joystick: {self.right_thumb[0]:.3f}, {self.right_thumb[1]:.3f}\n"+ 
                f"L/R Trigger: {self.left_trig:.2f}, {self.right_trig:.2f}   A: {self.a_held} X: {self.x_held} B: {self.b_held} Y: {self.y_held}")

    def output(self) -> list[float]:
        x = 200 * self.left_thumb[0]
        y = 200 * self.right_thumb[1]
        z = 325*(1 - 0.5*(self.left_trig+1)) 
        Rx = (math.pi/2) * self.right_thumb[0]
        Ry = (math.pi/2) * self.right_thumb[1]
        return [x, y, z, Rx, Ry, 0]
    
if __name__ == "__main__":
    controller = Controller()
    while True:
        controller.update()
        controller.display()
        time.sleep(0.1)
    
    
    
