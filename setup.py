from distutils.core import Extension, setup
from glob import glob


def main() -> None:
    c_files = [
        "myhttp.c",
        *glob("src/http/*.c"),
        *glob("src/util/*.c"),
    ]
    setup(
        name="MyHttp",
        version="1.0.0",
        description="Very simple Http client",
        ext_modules=[Extension("myhttp", c_files)],
    )


if __name__ == "__main__":
    main()
