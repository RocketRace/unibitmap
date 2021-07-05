import setuptools, re

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

with open("unibitmap/__init__.py") as fp:
    version = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", fp.read()).group(1)

with open("README.md") as fp:
    readme = fp.read()

setuptools.setup(
    name="unibitmap",
    description="A Python module & command-line interface for converting between RGB images and executable unicode art",
    url="https://github.com/RocketRace/unibitmap",
    packages=["unibitmap"],
    license="GPLv3",
    install_requires=requirements,
    version=version,
    long_description=readme,
)