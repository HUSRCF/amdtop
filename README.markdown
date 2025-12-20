AMDTOP
=====
(This project is forked from the NVTOP from Syllo which provide most of the component.)
What is NVTOP?
--------------

NVTOP stands for Neat Videocard TOP, a (h)top like task monitor for GPUs and
accelerators. It can handle multiple GPUs and print information about them in a
htop-familiar way.

Currently supported vendors are AMD (Linux amdgpu driver), Apple (limited M1 &
M2 support), Huawei (Ascend), Intel (Linux i915/Xe drivers), NVIDIA (Linux
proprietary divers), Qualcomm Adreno (Linux MSM driver), Broadcom VideoCore (Linux v3d driver).

Because a picture is worth a thousand words:

![NVTOP interface](/screenshot/NVTOP_ex1.png)

Table of Contents
-----------------

It has to be pointed out that the modified version had and only had tested on AMD GPU with following models: Radeon Pro W7900, Radeon RX7900XTX and Radeon RX590. Other models of cards hadn't been tested, if you had met any issues, feel free to leave at the _Issues_.

- [NVTOP Options and Interactive Commands](#nvtop-options-and-interactive-commands)
  - [Interactive Setup Window](#interactive-setup-window)
  - [Saving Preferences](#saving-preferences)
  - [NVTOP Manual and Command line Options](#nvtop-manual-and-command-line-options)
- [GPU Support](#gpu-support)
  - [AMD](#amd)
- [Build](#build)
- [Distribution Specific Installation Process](#distribution-specific-installation-process)
  - Unlike the offical package, here you might only git from source and compile it yourself, a detailed instruction would be given in the following session. Or you might refer to the offical [NVTOP Build](#nvtop-build), which should be the same.
- [NVTOP Build](#nvtop-build)
- [Troubleshoot](#troubleshoot)
- [License](#license)

NVTOP Options and Interactive Commands
--------------------------------------
### Interactive Setup Window

NVTOP has a builtin setup utility that provides a way to specialize the interface to your needs.
Simply press ``F2`` and select the options that are the best for you.

![NVTOP Setup Window](/screenshot/Nvtop-config.png)

### Saving Preferences

You can save the preferences set in the setup window by pressing ``F12``.
The preferences will be loaded the next time you run ``nvtop``.

### AMDTOP Manual and Command line Options

AMDTOP comes with a manpage!
```bash
man nvtop
```
For quick command line arguments help
```bash
amdtop -h
amdtop --help
```
⚠️A system linkage or systemtical path should be modified or added to make sure it could be call out with such command.
-----------

### AMD

NVTOP supports AMD GPUs using the `amdgpu` driver through the exposed DRM and
sysfs interface.

AMD introduced the fdinfo interface in kernel 5.14 ([browse kernel
source](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/gpu/drm/amd/amdgpu/amdgpu_fdinfo.c?h=linux-5.14.y)).
Hence, you will need a kernel with a version greater or equal to 5.14 to see the
processes using AMD GPUs.

Support for recent GPUs are regularly mainlined into the linux kernel, so please
use a recent-enough kernel for your GPU.

Build
-----

Several libraries are required in order for NVTOP to display GPU info:

* The *ncurses* library driving the user interface.
  * This makes the screen look beautiful.
* For NVIDIA: the *NVIDIA Management Library* (*NVML*) which comes with the GPU driver.
  * This queries the GPU for info.
* For AMD: the libdrm library used to query AMD GPUs through the kernel driver.
* For METAX: the *MetaX System Management Library* (*MXSML*) which comes with the GPU driver.
  * This queries the GPU for info.

## Distribution Specific Installation Process


## AMDTOP Build

```bash
git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=ON -DINTEL_SUPPORT=ON
make

# Install globally on the system
sudo make install

# Alternatively, install without privileges at a location of your choosing
# cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=ON -DINTEL_SUPPORT=ON -DCMAKE_INSTALL_PREFIX=/path/to/your/dir
# make
# make install
```

If you use **conda** as environment manager and encounter an error while building NVTOP, try `conda deactivate` before invoking `cmake`.

The build system supports multiple build types (e.g. -DCMAKE_BUILD_TYPE=RelWithDebInfo):

* Release: Binary without debug info
* RelWithDebInfo: Binary with debug info
* Debug: Compile with warning flags and address/undefined sanitizers enabled (for development purposes)

Troubleshoot
------------

- The plot looks bad:
  - Verify that you installed the wide character version of the ncurses library (libncurses**w**5-dev for Debian / Ubuntu), clean the build directory and restart the build process.
- **Putty**: Tell putty not to lie about its capabilities (`$TERM`) by setting the field ``Terminal-type string`` to ``putty`` in the menu
  ``Connection > Data > Terminal Details``.

License
-------

NVTOP is licensed under the GPLv3 license or any later version.
You will find a copy of the license inside the COPYING file of the repository or
at the GNU website <[www.gnu.org/licenses/](http://www.gnu.org/licenses/)>.
