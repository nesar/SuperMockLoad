"""Setup for SuperMockLoad.

Metadata is declared in pyproject.toml; this setup.py is a thin, standard shim
so classic workflows (`python setup.py ...`, older pip, editable installs) work
too.  Install with:  pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name='supermockload',
    version='0.1.0',
    description='Load and plot SPHEREx SuperMock lightcone catalogs '
                '(HACC Last Journey)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    python_requires='>=3.8',
    packages=find_packages(),                       # -> ['supermockload']
    include_package_data=True,                      # ships supermockload/data/**
    package_data={'supermockload': [
        'data/*.json',
        'data/*.npz',
        'data/observations/*',
        'data/filters/*',
    ]},
    install_requires=[
        'numpy>=1.20',
        'h5py>=3.0',
        'matplotlib>=3.4',
        'astropy>=5.0',
    ],
    extras_require={'notebooks': ['jupyter', 'ipykernel']},
)
