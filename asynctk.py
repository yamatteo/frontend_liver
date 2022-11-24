import asyncio
import inspect
import tkinter
from tkinter import Tk
import multiprocessing as mp

from .store import Store


class AsyncTk(Tk):
    def __init__(self):
        super().__init__()
        self.running = True
        self.tasks = []
        self.store = Store()

    async def tk_loop(self):
        active = None
        while self.running:
            if active is None:
                if self.tasks:
                    active = asyncio.create_task(self.tasks.pop(0))
                else:
                    self.update()
                    await asyncio.sleep(0.05)
            else:
                if active.done():
                    active = None
                else:
                    await asyncio.sleep(0.05)
                self.update()

    def stop(self):
        self.running = False

    def task_factory(self, f, *args, **kwargs):
        return lambda: self.add_task(get_co(f, *args, **kwargs))

    def add_task(self, f, *args, **kwargs):
        self.tasks.append(get_co(f, *args, *kwargs))

    def set_trigger(self, key, f):
        busykey = key + "__is_busy"

        def task_factory():
            if getattr(self.store, busykey) is False:
                setattr(self.store, busykey, True)

                async def co():
                    await get_co(f)
                    setattr(self.store, key, False)
                    setattr(self.store, busykey, False)

                self.add_task(co())

        self.store.new(busykey, value=False)
        self.store.new(key, value=True, callback=task_factory)

    def spawn_event_task(self, target, *args, **kwargs):
        async def atask():
            proc = mp.Process(target=target, args=args, kwargs=kwargs)
            proc.start()
            while proc.is_alive():
                await asyncio.sleep(0.05)
            self.store.redraw = True

        self.tasks.append(atask())


def get_co(f, *args, **kwargs):
    if inspect.iscoroutine(f) and not args and not kwargs:
        return f
    elif inspect.iscoroutinefunction(f):
        return f(*args, **kwargs)
    elif inspect.isfunction(f):
        async def _co():
            f(*args, kwargs)

        return _co()
    else:
        raise ValueError(f"Unexpected type(task)={type(f)} with args {args} and kwargs {kwargs}")
