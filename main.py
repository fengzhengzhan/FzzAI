from dependencies import *

def main():
    """
    FzzAI项目进入入口。
    The entry point for the FzzAI project.
    """
    # menu = ParamMenu()
    # menu.printLogo()
    # print(os.getcwd())
    logger = Logger("C:\Files\Data\Life\Code\GithubCode\FzzAI\dataset_autosave" + os.sep + confglobal.GLOBAL_LOG_PATH)
    logger.log("main test")

if __name__ == '__main__':
    main()