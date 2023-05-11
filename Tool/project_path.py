from dependencies import *


class ProjectPath:
    def __init__(self):
        __source_path = os.getcwd()
        __array_path = __source_path.split(os.sep)
        # print(__array_path)
        self.root_path = __array_path[0]
        for each in __array_path[1:]:
            self.root_path += os.sep + each
            if each == confglobal.ROOTDIR:
                break
        print("[+]{}".format(self.root_path))

        self.dateset_autosave_path = self.root_path + os.sep + confglobal.DATASET_AUTOSAVE_PATH
        print("[+]{}".format(self.dateset_autosave_path))


if __name__ == '__main__':
    ProjectPath()
