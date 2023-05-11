from dependencies import *


class Environment:
    def outputSceneManager(self):
        return ProcessTransportManager(ProcessReadScreen()).transport_manager

    def outputMemoryManager(self):
        pass
