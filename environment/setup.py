from setuptools import setup, find_packages

setup(
    name="autotrain",
    version="0.2.0",
    packages=["autotrain", "autotrain.tasks"],
    package_dir={"autotrain": "src", "autotrain.tasks": "src/tasks"},
)
