'''Prints the raw axes of joystick 0, to find which axis is which. Ctrl+C to stop.'''
import pygame, time
pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()
print(f"axes={js.get_numaxes()}  buttons={js.get_numbuttons()}  hats={js.get_numhats()}")
while True:
    pygame.event.pump()
    print([round(js.get_axis(i), 2) for i in range(js.get_numaxes())], end="\r")
    time.sleep(0.05)