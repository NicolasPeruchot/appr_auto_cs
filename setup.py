"""Manage dependencies."""
import pathlib

from typing import List

from setuptools import find_packages, setup


def _read(fname: str) -> str:
    with open(pathlib.Path(fname)) as fh:
        data = fh.read()
    return data


base_packages: List[str] = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
]

dev_packages = [
    "black",
    "ipykernel",
    "isort",
    "pre-commit",
]


setup(
    name="appr_auto_cs",
    version="0.0.1",
    packages=find_packages(),
    long_description=_read("README.md"),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
    },
)
