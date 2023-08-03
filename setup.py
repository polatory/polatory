import json
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, src_dir: str = "") -> None:
        super().__init__(name, sources=[])
        self.src_dir = os.fspath(Path(src_dir).resolve())


class CMakeBuild(build_ext):
    # https://stackoverflow.com/a/57883682/5614012
    @staticmethod
    def windows_buildenv():
        vswhere = Path(
            os.path.expandvars(
                "%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"
            )
        ).resolve()
        if not vswhere.exists():
            raise EnvironmentError("vswhere.exe not found at: %s", vswhere)

        try:
            vs_dir = Path(
                subprocess.run(
                    f'"{vswhere}" -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath',
                    stdout=subprocess.PIPE,
                    check=True,
                    text=True,
                ).stdout.rstrip()
            ).resolve()
        except subprocess.CalledProcessError:
            raise EnvironmentError("MSVC is not installed.")
        vcvars64 = (vs_dir / "VC/Auxiliary/Build/vcvars64.bat").resolve()

        output = subprocess.run(
            f'"{vcvars64}" && set', stdout=subprocess.PIPE, check=True, text=True
        ).stdout

        for line in output.splitlines():
            pair = line.split("=", 1)
            if len(pair) >= 2:
                os.environ[pair[0]] = pair[1]

    def build_extension(self, ext: CMakeExtension) -> None:
        toolchain_file = (
            Path.cwd() / "vcpkg/scripts/buildsystems/vcpkg.cmake"
        ).resolve()
        out_dir = (
            (Path.cwd() / self.get_ext_fullpath(ext.name)).parent / ext.name
        ).resolve()

        cmake_args = [
            "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={out_dir}",
            f"-DBUILD_PYTHON=ON",
            f"-DPOLATORY_VERSION={version}",
        ]

        if sys.platform == "darwin":
            brew_prefix = subprocess.run(
                ["brew", "--prefix"], stdout=subprocess.PIPE, check=True, text=True
            ).stdout.rstrip()
            cmake_args += [f"-DCMAKE_CXX_COMPILER={brew_prefix}/opt/llvm/bin/clang++"]
        elif sys.platform == "win32":
            self.windows_buildenv()

        build_temp_dir = (Path(self.build_temp) / ext.name).resolve()
        build_temp_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.src_dir, *cmake_args], cwd=build_temp_dir, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "_core"],
            cwd=build_temp_dir,
            check=True,
        )


with open(Path.cwd() / "vcpkg.json") as f:
    vcpkg_json = json.load(f)
    version = vcpkg_json["version"]

setup(
    version=version,
    ext_modules=[CMakeExtension("polatory")],
    cmdclass={"build_ext": CMakeBuild},
)
