from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='SubsampledGraphBasedDBSCAN',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("SubsampledGraphBasedDBSCAN",
                 sources=["subsampled_graph_based_dbscan.pyx"],
                 language="c++",
                 include_dirs=[numpy.get_include()])],
    author='Jennifer Jang',
    author_email='j.jang42@gmail.com'

)
