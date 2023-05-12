from dependencies import *

from Environment.change_env import ChangeEnv
from Environment.read_screen import ProcessReadScreen
from Environment.transport_manager import ProcessTransportManager

class Environment:
    def outputSceneManager(self):
        return ProcessTransportManager(ProcessReadScreen()).transport_manager

    def outputMemoryManager(self):
        pass

    def outputChangeEnv(self):
        return ChangeEnv()