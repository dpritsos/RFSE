import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setuptools.setup(
    name="RFSE",
    version="0.1.1",
    author="Dimitrios Pritsos",
    author_email="dpritsos@extremepro.gr",
    description="Random Feature Space Ensemble, Machine Learning Algorithm",
    long_description="",
    long_description_content_type="",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
)


ext_modules = [
    Extension(
        "RFSE/simimeasures/cy",
        ["RFSE/simimeasures/cy.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    setup_requires=[
        'setuptools>=18.0',
        'cython>=0.19.1',
    ],
    name='cy-parallel',
    ext_modules=cythonize(ext_modules),
)
