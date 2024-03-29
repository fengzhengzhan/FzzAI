import os

import conf as confglobal

__source_path = os.getcwd()
__array_path = __source_path.split(os.sep)
# print(__array_path)
root_path = __array_path[0]
for each in __array_path[1:]:
    root_path += os.sep + each
    if each == confglobal.ROOTDIR:
        break
# print(f"[*] Project Root Directory: {root_path}")

dateset_autosave_path = root_path + os.sep + confglobal.DATASET_AUTOSAVE_PATH
if not os.path.exists(dateset_autosave_path):
    os.makedirs(dateset_autosave_path)
    print(f"[+] Create File: {dateset_autosave_path}")

if __name__ == '__main__':
    print(root_path)
