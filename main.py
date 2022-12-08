import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", default=False)
args = parser.parse_args()

def main(main_class):
    w = main_class()
    w.mainloop()


if __name__ == "__main__":
    import importlib
    import sys
    from pathlib import Path

    args = parser.parse_args()

    this_file = Path(__file__)
    this_folder = this_file.parent
    sys.path.append(str(this_folder.parent))

    this_module = importlib.import_module(this_folder.name)
    this_module.connect(path=this_folder, mock=args.debug)
    main(this_module.MainWindow)
