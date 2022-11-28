import tkinter as tk
from multiprocessing import Value


class SharedBooleanVar(tk.Variable):
    """Value holder for boolean variables."""
    _default = False

    def __init__(self, master=None, value=False, name=None):
        """Construct a boolean variable.

        MASTER can be given as master widget.
        VALUE is an optional value (defaults to False)
        NAME is an optional Tcl name (defaults to PY_VARnum).

        If NAME matches an existing variable and VALUE is omitted
        then the existing value is retained.
        """
        super(SharedBooleanVar, self).__init__(master, value, name)
        self.shared = Value('i', int(value))

    def set(self, value):
        """Set the variable to VALUE."""
        try:
            with self.shared.get_lock():
                self.shared.value = int(value)
                ret = self._tk.globalsetvar(self._name, self._tk.getboolean(value))
        except AttributeError:
            ret = self._tk.globalsetvar(self._name, self._tk.getboolean(value))

        return ret

    initialize = set

    def get(self):
        """Return the value of the variable as a bool."""
        try:
            ret_shared = bool(self.shared.value)
            ret_local = self._tk.getboolean(self._tk.globalgetvar(self._name))
            assert ret_shared == ret_local
            return ret_shared
        except tk.TclError:
            raise ValueError("invalid literal for getboolean()")


class Store(object):
    def __init__(self):
        super(Store, self).__setattr__("d", dict())
        super(Store, self).__setattr__("callbacks", dict())

    def __getattr__(self, key):
        item = self.d[key]
        if isinstance(item, tk.Variable):
            return item.get()
        return item

    def __setattr__(self, key, value):
        try:
            if isinstance(self.d[key], tk.Variable):
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
            if isinstance(self.d[key], tk.Variable):
                self.d[key].trace_add("write", lambda *args: callback())
            else:
                self.callbacks[key].append(callback)
        except KeyError:
            self.d[key] = None
            self.callbacks[key] = [callback, ]
