from dependencies import *

from Environment.change_env import ChangeEnv
from Environment.read_screen import ProcessReadScreen
from Environment.transport_manager import ProcessTransportManager

class Environment:
    def outputSceneManager(self, region=(0, 0, 1280, 720), name_process=None):
        return ProcessTransportManager(ProcessReadScreen(region, name_process)).transport_manager

    def outputMemoryManager(self):
        pass

    def outputChangeEnv(self):
        return ChangeEnv()