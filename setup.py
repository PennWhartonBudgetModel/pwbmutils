"""Setup script for package.
"""

from setuptools import setup
import os

# defines __version__
exec(open("pwbmutils/version.py").read())

setup(
    name="pwbmutils",
    version=__version__,
    description="Collection of cross-component utility functions",
    url="https://github.com/PennWhartonBudgetModel/Utilities",
    author="Penn Wharton Budget Model",
    packages=["pwbmutils"],
    zip_safe=False,
    test_suite="nose.collector",
    install_requires=["luigi", "pandas", "portalocker"],
    test_requires=["nose"]
)
