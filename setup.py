from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='SubsampledNeighborhoodGraphDBSCAN',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("SubsampledNeighborhoodGraphDBSCAN",
                 sources=["subsampled_neighborhood_graph_dbscan.pyx"],
                 language="c++",
                 include_dirs=[numpy.get_include()])],
    author='Jennifer Jang',
    author_email='j.jang42@gmail.com'

)
