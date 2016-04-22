import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

#os.environ["CC"] = "clang++"
#os.environ["CXX"] = "clang++"


setup(name='lazygrad',
      version='1.0',
      description='',
      author='Tim Vieira and Ryan Cotterell',
      packages=['lazygrad'],
      install_requires=[
      ],
      include_dirs=[np.get_include(),
                    os.path.expanduser('~/anaconda/include/')],
      library_dirs = ['/usr/local/lib'],
      ext_modules = cythonize(['lazygrad/**/*.pyx']))
