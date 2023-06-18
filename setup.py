#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_namespace_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(os.path.join(HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# list of requirements
fusedrug_requirements = []
with open(os.path.join(HERE, "requirements/requirements.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            fusedrug_requirements.append(line.strip())

# list of requirements for fusedrug_examples
fusedrug_examples_requirements = []
with open(os.path.join(HERE, "fusedrug_examples/requirements.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            fusedrug_examples_requirements.append(line.strip())

# list of requirements for core packages for development
fuse_requirements_dev = []
with open(os.path.join(HERE, "requirements/requirements_dev.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            fuse_requirements_dev.append(line.strip())


# version
version_file = open(os.path.join(HERE, "VERSION.txt"))
version = version_file.read().strip()

setup(
    name="fuse-drug",
    version=version,
    description="drug discovery domain data, models, pipelines and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiomedSciAI/fuse-med-ml-drug/",
    author="IBM Research Israel Labs - Machine Learning for Healthcare and Life Sciences",
    author_email="alex.golts@ibm.com",
    packages=find_namespace_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="Apache License 2.0",
    install_requires=fusedrug_requirements,
    extras_require={"examples": fusedrug_examples_requirements, "dev": fuse_requirements_dev,},
)
