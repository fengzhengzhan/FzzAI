from dependencies import *


class AlgorithmReplayMemory:
    def __init__(self, path_autosave=projectpath.dateset_autosave_path):
        path_sqlitedb = os.path.join(path_autosave, confglobal.DATASET_SQLITE)
        if not os.path.exists(path_sqlitedb):
            os.makedirs(path_sqlitedb)
            projlog(INFO, "[+] " + str(path_sqlitedb))


if __name__ == '__main__':
    AlgorithmReplayMemory()