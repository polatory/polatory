Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. .\tools\Exec.ps1
. .\tools\Invoke-BatchFile.ps1

function buildenv() {
    $vswhere = Resolve-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    $vsDir = & $vswhere -latest `
        -requires Microsoft.VisualStudio.Workload.NativeDesktop `
        -property installationPath

    if (-not $vsDir) {
        throw 'Some of the required workloads/components of Visual Studio are not installed.'
    }

    Invoke-BatchFile "$vsDir\VC\Auxiliary\Build\vcvars64.bat"
}

if ($args.Length -lt 1) {
    throw 'No task is specified.'
}

Set-Location $PSScriptRoot

switch -regex ($args[0]) {
    '^init-vcpkg$' {
        Set-Location vcpkg
        Exec { .\bootstrap-vcpkg.bat }
        Exec { .\vcpkg remove --outdated --recurse }
        Exec { .\vcpkg install abseil boost-filesystem boost-program-options boost-serialization ceres double-conversion eigen3 flann gsl-lite gtest --triplet x64-windows }
        break
    }
    '^cmake$' {
        buildenv
        New-Item build -ItemType Directory -Force
        Set-Location build
        Exec { cmake .. -GNinja '-DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake' }
        break
    }
    '^b(uild)?$' {
        buildenv
        Set-Location build
        Exec { ninja }
        break
    }
    '^t(est)?$' {
        buildenv
        Set-Location build
        Exec { ctest -V }
        break
    }
    default {
        throw "Unrecognized task: $($args[0])"
    }
}
