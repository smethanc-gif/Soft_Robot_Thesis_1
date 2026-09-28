'''
Read input from an Xbox controller (or 8BitDo adapter presenting as one)
using the XInput-Python package, which wraps Windows' native XInput API
via ctypes (no compiled extension, so no build-tool requirements).

Requires: XInput-Python
    pip install XInput-Python
'''

import time
import XInput


def main():
    '''Prints the first connected controller's sticks, triggers and buttons at ~20 Hz until Ctrl+C.'''
    print("Waiting for a controller (0-3)... (Ctrl+C to stop)\n")

    connected = XInput.get_connected()
    player = next((i for i, c in enumerate(connected) if c), None)
    if player is None:
        print("No controller detected. Make sure it's connected/paired.")
        return

    print(f"Connected: controller {player}")
    print("Reading input...\n")

    try:
        while True:
            state = XInput.get_state(player)
            buttons = XInput.get_button_values(state)
            pressed = [name for name, held in buttons.items() if held]

            leftThumb, rightThumb = XInput.get_thumb_values(state)
            leftTrigger, rightTrigger = XInput.get_trigger_values(state)

            print(
                f"L-stick={leftThumb}  R-stick={rightThumb}  "
                f"L-trig={leftTrigger:.2f}  R-trig={rightTrigger:.2f}  "
                f"buttons={pressed}",
                end="\r",
            )
            time.sleep(0.05)  # ~20 Hz poll rate

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
