from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def config_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold(smiles: str) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=Chem.MolFromSmiles(smiles), includeChirality=False
    )


def fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    bitvect = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), radius, nBits=n_bits, useChirality=False,
        useFeatures=False,
    )
    out = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, out)
    return out


def rdkit2d(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    names = np.asarray([name for name, _ in Descriptors._descList], dtype=str)
    values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan, silent=True)
    return np.asarray([values[name] for name in names], dtype=np.float64), names


def load_processed(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def package_versions() -> dict[str, str]:
    names = ["PyTDC", "rdkit", "tabpfn", "scikit-learn", "xgboost", "lightgbm", "numpy",
             "pandas", "scipy", "matplotlib", "torch"]
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not-installed"
    return found


def hardware() -> dict:
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        cuda, gpu = False, None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": cuda,
        "gpu": gpu,
    }


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
