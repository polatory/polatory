# Building on Windows

1. Install Visual Studio 2022

   https://www.visualstudio.com/

   Under the **Workloads** tab, select the following item.

   - **Desktop & Mobile**

     - **Desktop development with C++**

   Under the **Individual components** tab, select the following item.

   - **Code tools**

     - **Git for Windows**

1. [Install Intel(R) oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

1. Clone Polatory

   ```pwsh
   git clone --recurse-submodules https://github.com/polatory/polatory.git
   cd polatory
   ```

   To update an existing repository:

   ```pwsh
   git pull --recurse-submodules
   ```

1. Build polatory

   ```pwsh
   .\run init-vcpkg
   .\run cmake
   .\run build
   ```
