from dependencies import *
from Environment.change_env import ChangeEnv

class Environment:
    def outputSceneManager(self):
        return ProcessTransportManager(ProcessReadScreen()).transport_manager

    def outputMemoryManager(self):
        pass

    def outputChangeEnv(self):
        return ChangeEnv()