# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='statistical-arbitrage',
    version='0.1.0',
    description='Simple pair trading on SP500 sets',
    long_description=readme,
    author='Natalie Bohyun Kim',
    author_email='kimbh43@gmail.com',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

