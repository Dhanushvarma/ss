import numpy as np
import time

from ss.gamepad.gamepad import available
from ss.gamepad.controllers import PS4


def get_gamepad_action(gamepad, position_gain=5e-4, orientation_gain=5e-4):
    """
    returns the action, control_active and terminate
    action : array of size 7, where first 6 elements are the position input and the last element is the gripper input
    control_active : boolean, True if the R2 key is pressed, False otherwise
    terminate : boolean, True if the CIRCLE key is pressed, False otherwise
    """

    action = np.zeros((7,))

    threshold = 1e-1  # to remove deadzone noise

    x = gamepad.axis("LEFT-Y") if abs(gamepad.axis("LEFT-Y")) > threshold else 0
    y = gamepad.axis("LEFT-X") if abs(gamepad.axis("LEFT-X")) > threshold else 0
    z = gamepad.axis("RIGHT-Y") if abs(gamepad.axis("RIGHT-Y")) > threshold else 0

    # position deltas
    action[0] = position_gain * x
    action[1] = position_gain * y
    action[2] = -position_gain * z

    # orientation deltas set to zero, TODO.
    action[3] = 0
    action[4] = 0
    action[5] = 0

    # gripper control
    action[6] = 1 if gamepad.axis("L2") == 1 else 0

    # terminate
    terminate = True if gamepad.beenPressed("CIRCLE") else False

    # optional : control active
    control_active = True if gamepad.axis("R2") == 1 else False

    return action, control_active, terminate


def connect_gamepad():
    if not available():
        print("Please connect your gamepad...")
        while not available():
            time.sleep(1.0)
    gamepad = PS4()
    print("Gamepad connected")
    gamepad.startBackgroundUpdates()
    return gamepad
