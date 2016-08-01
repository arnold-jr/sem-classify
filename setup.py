# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('License.txt') as f:
    license = f.read()

setup(
    name='sem-classify',
    version='0.0.1',
    description='Classify pixels of SEM images',
    long_description=readme,
    author='Joshua Arnold',
    author_email='j.arnold.111@gmail.com',
    url='https://github.com/arnold-jr/sem-classify',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

