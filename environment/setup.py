from setuptools import setup, find_packages

setup(
    name="AutoTrain",
    version="0.1.0",
    packages=["auto_train", "auto_train.tasks"],
    package_dir={"auto_train": "src", "auto_train.tasks": "src/tasks"},
)
