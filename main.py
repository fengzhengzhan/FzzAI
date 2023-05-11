from dependencies import *

def main():
    """
    FzzAI项目进入入口。
    The entry point for the FzzAI project.
    """
    menu = ParamMenu()
    menu.printLogo()
    print(os.path.dirname(__file__))

if __name__ == '__main__':
    main()