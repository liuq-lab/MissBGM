from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_version() -> str:
    version_ns: dict = {}
    exec(_read_text(ROOT / "missbgm" / "_version.py"), version_ns)
    return str(version_ns["__version__"])


setup(
    name="missbgm",
    version=_load_version(),
    description="Missingness-aware data imputation with Bayesian generative modeling (MissBGM).",
    long_description=_read_text(ROOT / "README.md"),
    long_description_content_type="text/markdown",
    author="liuq-lab",
    url="https://github.com/liuq-lab/MissBGM",
    project_urls={
        "Source": "https://github.com/liuq-lab/MissBGM",
        "Issues": "https://github.com/liuq-lab/MissBGM/issues",
        "README": "https://github.com/liuq-lab/MissBGM#readme",
    },
    license="MIT",
    license_files=["LICENSE"],
    python_requires=">=3.12",
    packages=find_packages(include=["missbgm", "missbgm.*"]),
    include_package_data=False,
    install_requires=[
        "numpy",
        "pandas",
        "pyyaml",
        "scikit-learn",
        "tensorflow>=2.18,<2.19",
        "tensorflow-probability==0.25.0",
        "tf-keras==2.18.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)

