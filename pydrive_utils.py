import functools
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union, Iterator, List

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFile

from .main import args

PathLike = Union[str, List[str], "AbstractPath", "DrivePath", "Path"]
drive = ...

class MockDrive:
    def ListFile(self, *args, **kwargs):
        return [self.CreateFile() for name in ["file1", "file2", "file3"]]

    def CreateFile(self, *args, **kwargs):
        return dict()


def setup(path: Path=None, debug=False):
    global drive
    if debug:
        print("Using mock drive.")
        drive = MockDrive()
        return
    if path is None:
        this_file = Path(__file__)
        path = this_file.parent
    gauth = GoogleAuth()
    gauth.settings["client_config_file"] = str(path / "client_secrets.json")
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

@dataclass
class AbstractPath:
    parts: list[str]

    def __lt__(self, other):
        return str(self) < str(other)

    def __repr__(self):
        return "AbstractPath(" + str(", ").join([repr(part) for part in self.parts]) + ")"

    def __str__(self):
        return str("/").join(self.parts)

    def __truediv__(self, sub: PathLike) -> "AbstractPath":
        sub = AbstractPath.__from__(sub)
        return AbstractPath(self.parts + sub.parts)

    @classmethod
    def __from__(cls, other: PathLike) -> "AbstractPath":
        if isinstance(other, list) and all(isinstance(x, str) for x in other):
            return cls(other)
        else:
            other = str(other)
            other = other.strip().strip("/")
            other = other.split("/")
            return cls(other)

    @property
    def ext(self):
        name, *exts = self.name.split(".")
        return str(".").join(exts)

    @property
    def name(self):
        if self.parts:
            return self.parts[-1]
        else:
            return ""

    @property
    def parent(self):
        return AbstractPath(self.parts[:-1])

    def relative_to(self, other: PathLike) -> "AbstractPath":
        other = AbstractPath.__from__(other)
        if self.parts[:len(other.parts)] == other.parts:
            return AbstractPath(self.parts[len(other.parts):])
        else:
            raise ValueError(f"{self} is not inside {other}")


@dataclass
class DrivePath(AbstractPath):
    root: str = "root"
    obj: GoogleDriveFile = field(default_factory=dict)

    def __eq__(self, other):
        if isinstance(other, DrivePath):
            return self.root == other.root and self.parts == other.parts
        else:
            return False

    def __repr__(self):
        return "DrivePath(" \
               + str(", ").join([repr(part) for part in self.parts]) \
               + (", " if self.parts else "") \
               + f"root={repr(self.root)}," \
               + f"obj={repr(self.obj)})"

    def __str__(self):
        return "/" + str("/").join(self.parts)

    def __truediv__(self, sub: PathLike) -> "DrivePath":
        if isinstance(sub, DrivePath):
            assert self.root == sub.root
            parts = self.parts + sub.relative_to(self).parts
            obj = sub.obj
        else:
            parts = self.parts + AbstractPath.__from__(sub).parts
            obj = {}
        return DrivePath(parts=parts, root=self.root, obj=obj)

    @property
    def id(self):
        self.resolve()
        return self.obj.get("id", None)

    @property
    def parent(self):
        return DrivePath(self.parts[:-1], root=self.root, obj=drive.CreateFile())

    def exists(self):
        try:
            self.resolve()
            return True
        except (AssertionError, FileNotFoundError):
            return False

    def iterdir(self):
        self.resolve()
        for obj in drive.ListFile({'q': f"'{self.id}' in parents and trashed=false"}).GetList():
            yield DrivePath(self.parts + [obj.metadata["title"]], root=self.root, obj=obj)

    def is_dir(self):
        self.resolve()
        return self.obj["mimeType"] == "application/vnd.google-apps.folder"

    def mkdir(self, exists_ok=True, make_parents=True):
        if self.exists():
            if not exists_ok:
                raise ValueError(f"Path {self} is already existing.")
        else:
            parent = self.parent
            if make_parents:
                parent.mkdir()
            drive.CreateFile(
                dict(
                    title=self.name,
                    parents=[{"id": parent.id}],
                    mimeType='application/vnd.google-apps.folder'
                )).Upload()
            self.resolve()
        return self

    def resolve(self):
        if "id" in self.obj and "metadata" in self.obj:
            return self
        obj = get_item(item_id=self.root)
        for i, title in enumerate(self.parts):
            obj = get_item(title=title, parent_id=obj["id"])
        self.obj = obj
        return self


def list_items(title=None, parent_id=None, parent_folder=None, is_folder=False):
    query = []
    if title:
        query.append(f"title='{title}'")
    if parent_id:
        query.append(f"'{parent_id}' in parents")
    elif parent_folder:
        query.append(f"'{parent_folder['id']}' in parents")
    if is_folder:
        query.append("mimeType='application/vnd.google-apps.folder'")
    query.append("trashed=false")
    query = str(" and ").join(query)
    return drive.ListFile({'q': query}).GetList()


def get_item(item_id=None, **kwargs):
    if item_id:
        return drive.CreateFile({"id": item_id})
    items_list = list_items(**kwargs)
    assert len(items_list) == 1
    return items_list[0]


# Criteria
def contains(path: DrivePath, filelist: list) -> bool:
    """True if path contains all listed files."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return all(name in files for name in filelist)


def is_anything(path: DrivePath) -> bool:
    """True if path contains something related to this project."""
    return is_dicom(path) or is_original(path) or is_registered(path) or is_trainable(path)


def is_dicom(path: DrivePath) -> bool:
    """True if path contains DICOMDIR."""
    return contains(path, ["DICOMDIR"])


def is_original(path: DrivePath) -> bool:
    """True if path contains original nifti scans."""
    return contains(path, [f"original_phase_{phase}.nii.gz" for phase in ["b", "a", "v", "t"]])


def is_registered(path: DrivePath) -> bool:
    """True if path contains registered nifti scans."""
    return contains(path, [f"registered_phase_{phase}.nii.gz" for phase in ["b", "a", "v", "t"]])


def is_predicted(path: DrivePath) -> bool:
    """True if path contains prediction."""
    return contains(path, ["prediction.nii.gz"])


def is_trainable(path: DrivePath) -> bool:
    """True if path contains segmentation and registered nifti scans."""
    return is_registered(path) and contains(path, ["segmentation.nii.gz"])


# Discover utility
def discover(path: DrivePath, select_dir: Callable = is_anything) -> list[DrivePath]:
    """Recursively list dirs in `path` that respect `select_dir` criterion."""
    unexplored_paths = [path]
    selected_paths = []
    while len(unexplored_paths) > 0:
        new_path = unexplored_paths.pop(0)
        if select_dir(new_path):
            selected_paths.append(new_path)
        elif new_path.is_dir():
            unexplored_paths.extend(new_path.iterdir())
    selected_paths.sort()
    return selected_paths


# Iterators
def iter_containing(path: DrivePath, filelist: list) -> Iterator[DrivePath]:
    """Iterates over subfolders containing listed files."""
    yield from discover(path, functools.partial(contains, filelist=filelist))


def iter_dicom(path: DrivePath) -> Iterator[DrivePath]:
    """Iterates over DICOMDIR subfolders."""
    yield from discover(path, is_dicom)


def iter_original(path: DrivePath) -> Iterator[DrivePath]:
    """Iterates over subfolders containing original nifti scans."""
    yield from discover(path, is_original)


def iter_registered(path: DrivePath) -> Iterator[DrivePath]:
    """Iterates over subfolders containing registered nifti scans."""
    yield from discover(path, is_registered)


def iter_trainable(path: DrivePath) -> Iterator[DrivePath]:
    """Iterates over subfolders containing registered nifti scans and segmentations."""
    yield from discover(path, is_trainable)


def split_trainables(path: DrivePath, n: int = 10, shuffle=False, offset=0) -> tuple[list[DrivePath], list[DrivePath]]:
    """Lists of case_path for train and valid dataset."""
    trainables = list(iter_trainable(path))
    if offset:
        trainables = [*trainables[offset:], *trainables[:offset]]
    train_cases = list(path / case for k, case in enumerate(trainables) if k % n != 0)
    valid_cases = list(path / case for k, case in enumerate(trainables) if k % n == 0)
    if shuffle:
        train_cases = random.sample(train_cases, len(train_cases))
    return train_cases, valid_cases
