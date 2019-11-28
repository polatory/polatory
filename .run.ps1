Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# https://github.com/Pscx/Pscx/blob/81b76cfdb1343f84880e0e2cd647db5c56cf354b/Src/Pscx/Modules/Utility/Pscx.Utility.psm1#L745-L767
function Invoke-BatchFile
{
    param([string]$Path, [string]$Parameters)

    $tempFile = [IO.Path]::GetTempFileName()

    ## Store the output of cmd.exe.  We also ask cmd.exe to output
    ## the environment table after the batch file completes
    cmd.exe /c " `"$Path`" $Parameters && set " > $tempFile

    ## Go through the environment variables in the temp file.
    ## For each of them, set the variable in our local environment.
    Get-Content $tempFile | Foreach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Content "env:\$($matches[1])" $matches[2]
        }
        else {
            $_
        }
    }

    Remove-Item $tempFile
}

function buildenv() {
    $vswhere = Resolve-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    # VS 2017 15.3 is the first version which ships with Ninja.
    $vsDir = & $vswhere -version "[15.3,17.0)" `
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

switch ($args[0]) {
    'build' {
        buildenv
        Set-Location build
        & ninja
        break
    }
    'cmake' {
        buildenv
        New-Item build -ItemType Directory -Force
        Set-Location build
        & cmake .. -GNinja '-DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake'
        break
    }
    'init-vcpkg' {
        Set-Location vcpkg
        .\bootstrap-vcpkg.bat
        .\vcpkg remove --outdated --recurse
        .\vcpkg install abseil boost-filesystem boost-program-options boost-serialization ceres double-conversion eigen3 flann gsl-lite gtest intel-mkl --triplet x64-windows
        break
    }
    'test' {
        buildenv
        Set-Location build
        & ctest -V
        break
    }
    default {
        throw "Unrecognized task: $($args[0])"
    }
}
