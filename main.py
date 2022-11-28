import argparse
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", default=False)
args = SimpleNamespace(
    debug=False
)

def main(build):
    root = build()
    root.mainloop()


if __name__ == "__main__":
    import importlib
    import sys
    from pathlib import Path
    args = parser.parse_args()


    this_file = Path(__file__)
    this_folder = this_file.parent
    sys.path.append(str(this_folder.parent))

    this_module = importlib.import_module(this_folder.name)
    this_module.pu.setup(path=this_folder, debug=args.debug)
    main(this_module.build_root)
