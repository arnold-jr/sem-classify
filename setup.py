# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sem_classify',
    version='0.0.1',
    description='Classify pixels of SEM images',
    long_description=readme,
    author='Joshua Arnold',
    author_email='j.arnold.111@gmail.com',
    url='https://github.com/arnold-jr/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

