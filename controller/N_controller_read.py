"""
Read input from an Xbox controller (or 8BitDo adapter presenting as one)
using the XInput-Python package, which wraps Windows' native XInput API
via ctypes (no compiled extension, so no build-tool requirements).

Requires: XInput-Python
    pip install XInput-Python
"""

import time
import XInput


def main():
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

            left_thumb, right_thumb = XInput.get_thumb_values(state)
            left_trigger, right_trigger = XInput.get_trigger_values(state)

            print(
                f"L-stick={left_thumb}  R-stick={right_thumb}  "
                f"L-trig={left_trigger:.2f}  R-trig={right_trigger:.2f}  "
                f"buttons={pressed}",
                end="\r",
            )
            time.sleep(0.05)  # ~20 Hz poll rate

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
