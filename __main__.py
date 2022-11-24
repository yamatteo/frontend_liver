import asyncio

from . import build_root

root = build_root()
root.update()
asyncio.run(root.tk_loop())