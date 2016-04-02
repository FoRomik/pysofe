
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import os
import sys

import pysofe

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', os.linesep)
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return linesep.join(buf)

long_description = read('README.rst')

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    name = 'pysofe',
    version = pysofe.__version__,
    url = 'https://github.com/and2345/pysofe',
    license='BSD License',
    author='Andreas Kunze',
    install_requires=['numpy>=1.10.4',
                    'scipy>=0.17.0',
                    ],
    tests_require=['pytest'],
    extras_require={
        'testing': ['pytest'],
        'visualization' : ['matplotlib>=1.5.1'],
    cmdclass={'test': PyTest},
    author_email='andreas.kunze@mailbox.tu-dresden.de',
    description='Finite element method software package for solving partial differential equations in 1D, 2D and 3D',
    long_description=long_description,
    packages=['pysofe'],
    platforms='any',
    classifiers = [
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Environment :: Console',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research'
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        ],
    }
)
