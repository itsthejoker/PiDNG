# many thanks to https://stackoverflow.com/a/60163996
from distutils.command.build_ext import build_ext
from distutils.errors import DistutilsPlatformError, DistutilsExecError, CCompilerError

from setuptools import Extension


ext_modules = [
    Extension(
        'ljpegCompress',
        sources=[
            "pidng/bitunpack.c", "pidng/liblj92/lj92.c"
        ],
        extra_compile_args=['-std=gnu99'],
        extra_link_args=[]
    )
]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed('File not found. Could not compile C extension.')

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed('Could not compile C extension.')


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": ExtBuilder}}
    )
