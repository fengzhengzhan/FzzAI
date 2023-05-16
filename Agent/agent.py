from dependencies import *

from action import Action
from listener import ProcessListenerKeyboard
from Environment.transport_manager import ProcessTransportManager


class Agent:
    def inputAction(self, has_keyboard, has_mouse):
        return Action(has_keyboard, has_mouse)

    def inputListenerManager(self):
        plk = ProcessListenerKeyboard()
        listener_manager = ProcessTransportManager(plk).transport_manager
        map_keycode = plk.map_keycode
        return (listener_manager, map_keycode)
