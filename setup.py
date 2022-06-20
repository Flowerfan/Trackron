import glob
import os
from os import path
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"


def get_version():
  init_py_path = path.join(path.abspath(path.dirname(__file__)), "trackron",
                           "__init__.py")
  init_py = open(init_py_path, "r").readlines()
  version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
  version = version_line.split("=")[-1].strip().strip("'\"")
  return version


def get_extensions():
  this_dir = path.dirname(path.abspath(__file__))
  extensions_dir = path.join(this_dir, "trackron", "external", "csrc")

  main_source = path.join(extensions_dir, "vision.cpp")
  sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

  from torch.utils.cpp_extension import ROCM_HOME

  is_rocm_pytorch = (True if ((torch.version.hip is not None) and
                              (ROCM_HOME is not None)) else False)
  if is_rocm_pytorch:
    assert torch_ver >= [1, 7], "ROCM support requires PyTorch >= 1.7!"

  # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
  source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
      path.join(extensions_dir, "*.cu"))
  sources = [main_source] + sources

  extension = CppExtension

  extra_compile_args = {"cxx": []}
  define_macros = []

  if (torch.cuda.is_available() and
      ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
          "FORCE_CUDA", "0") == "1":
    extension = CUDAExtension
    sources += source_cuda

    if not is_rocm_pytorch:
      define_macros += [("WITH_CUDA", None)]
      extra_compile_args["nvcc"] = [
          "-O3",
          "-DCUDA_HAS_FP16=1",
          "-D__CUDA_NO_HALF_OPERATORS__",
          "-D__CUDA_NO_HALF_CONVERSIONS__",
          "-D__CUDA_NO_HALF2_OPERATORS__",
      ]
    else:
      define_macros += [("WITH_HIP", None)]
      extra_compile_args["nvcc"] = []

    if torch_ver < [1, 7]:
      CC = os.environ.get("CC", None)
      if CC is not None:
        extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

  include_dirs = [extensions_dir]

  ext_modules = [
      extension(
          "trackron._C",
          sources,
          include_dirs=include_dirs,
          define_macros=define_macros,
          extra_compile_args=extra_compile_args,
      )
  ]

  return ext_modules


setup(
    name="trackron",
    version=get_version(),
    author="flowerfan",
    url="https://github.com/flowerfan/trackron",
    description="trackron is platform for object tracking",
    packages=find_packages(exclude=("configs")),
    python_requires=">=3.6",
    install_requires=[
        # These dependencies are not pure-python.
        # In general, avoid adding more dependencies like them because they are not
        # guaranteed to be installable by `pip install` on all platforms.
        # To tell if a package is pure-python, go to https://pypi.org/project/{name}/#files
        "Pillow>=7.1",  # or use pillow-simd for better performance
        "matplotlib",  # TODO move it to optional after we add opencv visualization
        "pycocotools>=2.0.2",  # corresponds to https://github.com/ppwwyyxx/cocoapi
        # Do not add opencv here. Just like pytorch, user should install
        # opencv themselves, preferrably by OS's package manager, or by
        # choosing the proper pypi package name at https://github.com/skvark/opencv-python
        # The following are pure-python dependencies that should be easily installable
        "termcolor>=1.1",
        "yacs>=0.1.8",
        "tabulate",
        "cloudpickle",
        "tqdm>4.29.0",
        "tensorboard",
        # Lock version of fvcore/iopath because they may have breaking changes
        # NOTE: when updating fvcore/iopath version, make sure fvcore depends
        # on compatible version of iopath.
        "fvcore>=0.1.5,<0.1.6",  # required like this to make it pip installable
        "iopath>=0.1.7,<0.1.10",
        "dataclasses; python_version<'3.7'",
        "omegaconf>=2.1",
        "hydra-core>=1.1",
        "black==21.4b2",
        "scipy>1.5.1",
        # If a new dependency is required at import time (in addition to runtime), it
        # probably needs to exist in docs/requirements.txt, or as a mock in docs/conf.py
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)