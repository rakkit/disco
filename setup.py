from setuptools import setup, find_packages

setup(
    name="disco",
    version="0.0.1",
    description="Disco: Distributed Spectral Condition Optimizer",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[],
    author="J. Wang, O. Filatov, J.Ebert, S. Kesselheim",
)
