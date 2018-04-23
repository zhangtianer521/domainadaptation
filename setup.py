from setuptools import setup, find_packages
from codecs import open
from os import path

import meanseg

here = path.abspath(path.dirname(__file__))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='meanseg',
    version=meanseg.__version__,
    description='Segmentation domain adaptation for MRI',
    url='https://github.com/neuropoly/domainadaptation',
    author='Neuropoly and GPIN',
    author_email='email@email.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=requirements,
    #entry_points={
    #    'console_scripts': [
    #        'domainadapt=domainadapt.main:run_main',
    #    ],
    #},
)
