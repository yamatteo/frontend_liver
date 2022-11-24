from __future__ import annotations

import pickle
from pathlib import Path

import nibabel
import numpy as np


def load_ndarray(file_path: Path) -> np.ndarray:
    image = nibabel.load(file_path)
    return np.array(image.dataobj, dtype=np.int16)


def save_segmentation(segm: np.ndarray, case_path: Path):
    affine, _, _, _ = load_registration_data(case_path)
    nibabel.save(
        nibabel.Nifti1Image(
            segm,
            affine=affine
        ),
        case_path / "segmentation.nii.gz",
    )


def load_registration_data(case_path: Path) -> tuple[np.ndarray, int, int, int]:
    with open(case_path / "registration_data.pickle", "rb") as f:
        d = pickle.load(f)
    return d["affine"], d["bottom"], d["top"], d["height"]


def load(case_path: Path, scan: bool = True, segm: bool = False, clip: tuple[int, int] = None) -> dict:
    print(f"Loading {case_path}...")
    name = str(case_path.name)
    _, bottom, top, _ = load_registration_data(case_path)
    if scan:
        scan = np.stack([
            load_ndarray(case_path / f"registered_phase_{phase}.nii.gz")
            for phase in ["b", "a", "v", "t"]
        ])
        scan = scan[..., bottom:top]
        if clip:
            np.clip(scan, *clip, out=scan)
        scan = scan.astype(np.float32)
    else:
        scan = None

    try:
        assert segm
        segm = load_ndarray(case_path / f"segmentation.nii.gz")
        assert np.all(segm < 3), "Segmentation has indices above 2."
        segm = segm[..., bottom:top]
        segm = segm.astype(np.int64)
    except (FileNotFoundError, AssertionError) as err:
        print("Error loading segmentation.", err)
        segm = None

    return dict(scan=scan, segm=segm, name=name)
