import tkinter


class Store(object):
    def __init__(self):
        super(Store, self).__setattr__("d", dict())
        super(Store, self).__setattr__("callbacks", dict())

    def __getattr__(self, key):
        item = self.d[key]
        if isinstance(item, tkinter.Variable):
            return item.get()
        return item

    def __setattr__(self, key, value):
        try:
            if isinstance(self.d[key], tkinter.Variable):
                self.d[key].set(value)
            else:
                self.d[key] = value
            for f in self.callbacks[key]:
                f()
        except KeyError:
            self.d[key] = value
            self.callbacks[key] = []

    def new(self, key, value=None, callback=None):
        if key in self.d:
            raise KeyError(f"Key {key} is not new.")
        self.d[key] = value
        self.callbacks[key] = []
        if callback:
            callback()
            self.trace_add(key, callback)

        return value

    def trace_add(self, key, callback):
        try:
            if isinstance(self.d[key], tkinter.Variable):
                self.d[key].trace_add("write", lambda *args: callback())
            else:
                self.callbacks[key].append(callback)
        except KeyError:
            self.d[key] = None
            self.callbacks[key] = [callback, ]
