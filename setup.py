#!/usr/bin/env python
from setuptools import setup, find_packages
import re
import os


def get_version():
    fn = os.path.join(os.path.dirname(__file__), "eli5", "__init__.py")
    with open(fn) as f:
        return re.findall("__version__ = '([\d.\w]+)'", f.read())[0]


def get_long_description():
    readme = open('README.rst').read()
    changelog = open('CHANGES.rst').read()
    return "\n\n".join([
        readme,
        changelog.replace(':func:', '').replace(':ref:', '')
    ])

setup(
    name='eli5',
    version=get_version(),
    author='Mikhail Korobov, Konstantin Lopuhin',
    author_email='kmike84@gmail.com, kostia.lopuhin@gmail.com',
    license='MIT license',
    long_description=get_long_description(),
    description="Debug machine learning classifiers and explain their predictions",
    url='https://github.com/eli5-org/eli5',
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'attrs > 17.1.0',
        'jinja2 >= 3.0.0',
        'numpy >= 1.9.0',
        'scipy',
        'scikit-learn >= 1.6.0',
        'graphviz',
        'tabulate>=0.7.7',
    ],
    python_requires=">=3.9",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
