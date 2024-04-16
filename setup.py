from distutils.core import Extension, setup
from glob import glob
import numpy


def main() -> None:
    c_files = [
        "example.c",
    ]
    setup(
        name="MyExample",
        version="1.0.12",
        description="Very simple Python C extension example",
        ext_modules=[Extension("myexample", c_files, include_dirs=[numpy.get_include()])],
        install_requires=['numpy']
    )


if __name__ == "__main__":
    main()
