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
    '^cmake$' {
        buildenv
        Exec { cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE='vcpkg/scripts/buildsystems/vcpkg.cmake' }
        break
    }
    '^b(uild)?$' {
        buildenv
        Exec { cmake --build build }
        break
    }
    '^t(est)?$' {
        buildenv
        Exec { ctest -V --test-dir build }
        break
    }
    default {
        throw "Unrecognized task: $($args[0])"
    }
}
