"""Setup script for package.
"""

from setuptools import setup
import os

setup(
    name="pwbmutils",
    version="0.98",
    description="Collection of cross-component utility functions",
    url="https://github.com/PennWhartonBudgetModel/Utilities",
    author="Penn Wharton Budget Model",
    packages=["pwbmutils"],
    zip_safe=False,
    test_suite="nose.collector",
    install_requires=["luigi", "pandas", "portalocker"],
    test_requires=["nose"]
)
