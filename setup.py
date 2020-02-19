import re
from codecs import open
from pathlib import Path

from setuptools import setup, find_packages

here = Path.resolve(Path(__file__).parent)


def read(*parts):
    with open(here.joinpath(*parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


with open(here / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='candlelight',
    version=find_version('candlelight', '__init__.py'),
    description='Collection of PyTorch layers modelling functions through splines',
    long_description=long_description,
    author='Nextbit AI Team',
    author_email='lgrementieri@nextbit.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Data scientists',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Apache 2.0',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    keywords='pytorch layer function approximation spline',
    packages=find_packages(exclude=['build', 'dist', 'docs', 'tests']),
    python_requires='>=3.6',
    install_requires=[
        'torch >= 1.4',
    ],
)
