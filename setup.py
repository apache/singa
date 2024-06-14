#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
'''Script for building wheel package for installing singa via pip.

This script must be launched at the root dir of the singa project
inside the docker container created via tool/docker/devel/centos/cudaxx/Dockerfile.manylinux2014.

    # launch docker container
    $ nvidia-docker run -v <local singa dir>:/root/singa -it apache/singa:manylinux2014-cuda10.2
    # build the wheel packag; replace cp36-cp36m to compile singa for other py version
    $ /opt/python/cp36-cp36m/bin/python setup.py bdist_wheel
    $ /opt/python/cp37-cp37m/bin/python setup.py bdist_wheel
    $ /opt/python/cp38-cp38/bin/python setup.py bdist_wheel

The generted wheel file should be repaired by the auditwheel tool to make it
compatible with PEP513. Otherwise, the dependent libs will not be included in
the wheel package and the wheel file will be rejected by PYPI website during
uploading due to file name error.

    # repair the wheel pakage and upload to pypi
    $ /opt/python/cp36-cp36m/bin/python setup.py audit

For the Dockerfile with CUDA and CUDNN installed, the CUDA version and
CUDNN version are exported as environment variable: CUDA_VERSION, CUDNN_VERSION.
You can control the script to build CUDA enabled singa package by exporting
SINGA_CUDA=ON; otherwise the CPU only package will be built.


Ref:
[1] https://github.com/bytedance/byteps/blob/master/setup.py
[2] https://setuptools.readthedocs.io/en/latest/setuptools.html
[3] https://packaging.python.org/tutorials/packaging-projects/
'''

from setuptools import find_packages, setup, Command, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsSetupError

import os
import io
import sys
import subprocess
import shutil
import shlex
from pathlib import Path

import numpy as np

NAME = 'singa'
'''
Pypi does not allow you to overwrite the uploaded package;
therefore, you have to bump the version.
Pypi does not allow [local version label](https://www.python.org/dev/peps/pep-0440/#local-version-segments)
to appear in the version, therefore, you have to include the public
version label only. Currently, due to the pypi size limit, the package
uploaded to pypi is cpu only (without cuda and cudnn), which can be installed via

    $ pip install singa
    $ pip install singa=3.0.0.dev1

The cuda and cudnn enabled package's version consists of the public
version label + local version label, e.g., 3.0.0.dev1+cuda10.2, which
can be installed via

    $ pip install singa=3.0.0.dev1+cuda10.2 -f <url of the repo>

'''
from datetime import date

# stable version
VERSION = '4.2.0'
# get the git hash
# git_hash = subprocess.check_output(["git", "describe"]).strip().split('-')[-1][1:]
# comment the next line to build wheel for stable version
# VERSION += '.dev' + date.today().strftime('%y%m%d')

SINGA_PY = Path('python')
SINGA_SRC = Path('src')
SINGA_HDR = Path('include')


class AuditCommand(Command):
    """Support setup.py upload."""

    description = 'Repair the package via auditwheel tool.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.status('Removing previous wheel files under wheelhouse')
        shutil.rmtree('wheelhouse', ignore_errors=True)
        for wheel in os.listdir('dist'):
            self.status('Repair the dist/{} via auditwheel'.format(wheel))
            os.system('auditwheel repair dist/{}'.format(wheel))

        # self.status('Uploading the package to PyPI via Twine…')
        # os.system('{} -m twine upload dist/*'.format(sys.executable))
        sys.exit()


def parse_compile_options():
    '''Read the environment variables to parse the compile options.

    Returns:
        a tuple of bool values as the indicators
    '''
    with_cuda = os.environ.get('SINGA_CUDA', False)
    with_nccl = os.environ.get('SINGA_NCCL', False)
    with_test = os.environ.get('SINGA_TEST', False)
    with_debug = os.environ.get('SINGA_DEBUG', False)

    return with_cuda, with_nccl, with_test, with_debug


def generate_singa_config(with_cuda, with_nccl):
    '''Generate singa_config.h file to define some macros for the cpp code.

    Args:
        with_cuda(bool): indicator for cudnn and cuda lib
        with_nccl(bool): indicator for nccl lib
    '''
    config = ['#define USE_CBLAS', '#define USE_GLOG', '#define USE_DNNL']
    if not with_cuda:
        config.append('#define CPU_ONLY')
    else:
        config.append('#define USE_CUDA')
        config.append('#define USE_CUDNN')

    if with_nccl:
        config.append('#define ENABLE_DIST')
        config.append('#define USE_DIST')

    # singa_config.h to be included by cpp code
    cpp_conf_path = SINGA_HDR / 'singa/singa_config.h'
    print('Writing configs to {}'.format(cpp_conf_path))
    with cpp_conf_path.open('w') as fd:
        for line in config:
            fd.write(line + '\n')
        versions = [int(x) for x in VERSION.split('+')[0].split('.')[:3]]
        fd.write('#define SINGA_MAJOR_VERSION {}\n'.format(versions[0]))
        fd.write('#define SINGA_MINOR_VERSION {}\n'.format(versions[1]))
        fd.write('#define SINGA_PATCH_VERSION {}\n'.format(versions[2]))
        fd.write('#define SINGA_VERSION "{}"\n'.format(VERSION))

    # config.i to be included by swig files
    swig_conf_path = SINGA_SRC / 'api/config.i'
    with swig_conf_path.open('w') as fd:
        for line in config:
            fd.write(line + ' 1 \n')

        fd.write('#define USE_PYTHON 1\n')
        if not with_nccl:
            fd.write('#define USE_DIST 0\n')
        if not with_cuda:
            fd.write('#define USE_CUDA 0\n')
            fd.write('#define USE_CUDNN 0\n')
        else:
            fd.write('#define CUDNN_VERSION "{}"\n'.format(
                os.environ.get('CUDNN_VERSION')))
        versions = [int(x) for x in VERSION.split('+')[0].split('.')[:3]]
        fd.write('#define SINGA_MAJOR_VERSION {}\n'.format(versions[0]))
        fd.write('#define SINGA_MINOR_VERSION {}\n'.format(versions[1]))
        fd.write('#define SINGA_PATCH_VERSION {}\n'.format(versions[2]))
        fd.write('#define SINGA_VERSION "{}"\n'.format(VERSION))


def get_cpp_flags():
    default_flags = ['-std=c++11', '-fPIC', '-g', '-O2', '-Wall', '-pthread']
    # avx_flags = [ '-mavx'] #'-mf16c',
    if sys.platform == 'darwin':
        # Darwin most likely will have Clang, which has libc++.
        return default_flags + ['-stdlib=libc++']
    else:
        return default_flags


def generate_proto_files():
    print('----------------------')
    print('Generating proto files')
    print('----------------------')
    proto_src = SINGA_SRC / 'proto'
    cmd = "/usr/bin/protoc --proto_path={} --cpp_out={} {}".format(
        proto_src, proto_src, proto_src / 'core.proto')
    subprocess.run(cmd, shell=True, check=True)

    proto_hdr_dir = SINGA_HDR / 'singa/proto'
    proto_hdr_file = proto_hdr_dir / 'core.pb.h'
    if proto_hdr_dir.exists():
        if proto_hdr_file.exists():
            proto_hdr_file.unlink()
    else:
        proto_hdr_dir.mkdir()

    shutil.copyfile(Path(proto_src / 'core.pb.h'), proto_hdr_file)
    return proto_hdr_file, proto_src / 'core.pb.cc'


def path_to_str(path_list):
    return [str(x) if not isinstance(x, str) else x for x in path_list]


def prepare_extension_options():
    with_cuda, with_nccl, with_test, with_debug = parse_compile_options()

    generate_singa_config(with_cuda, with_nccl)
    generate_proto_files()

    link_libs = ['glog', 'protobuf', 'openblas', 'dnnl']

    sources = path_to_str([
        *list((SINGA_SRC / 'core').rglob('*.cc')), *list(
            (SINGA_SRC / 'model/operation').glob('*.cc')), *list(
                (SINGA_SRC / 'utils').glob('*.cc')),
        SINGA_SRC / 'proto/core.pb.cc', SINGA_SRC / 'api/singa.i'
    ])
    include_dirs = path_to_str([
        SINGA_HDR, SINGA_HDR / 'singa/proto',
        np.get_include(), '/usr/include', '/usr/include/openblas',
        '/usr/local/include'
    ])

    try:
        np_include = np.get_include()
    except AttributeError:
        np_include = np.get_numpy_include()
    include_dirs.append(np_include)

    library_dirs = []  # path_to_str(['/usr/lib64', '/usr/local/lib'])

    if with_cuda:
        link_libs.extend(['cudart', 'cudnn', 'curand', 'cublas', 'cnmem'])
        include_dirs.append('/usr/local/cuda/include')
        library_dirs.append('/usr/local/cuda/lib64')
        sources.append(str(SINGA_SRC / 'core/tensor/math_kernel.cu'))
        if with_nccl:
            link_libs.extend(['nccl', 'cusparse', 'mpicxx', 'mpi'])
            sources.append(str(SINGA_SRC / 'io/communicator.cc'))
    # print(link_libs, extra_libs)

    libraries = link_libs
    runtime_library_dirs = ['.'] + library_dirs
    extra_compile_args = {'gcc': get_cpp_flags()}

    if with_cuda:
        # compute_35 and compute_50 are removed because 1. they do not support half float;
        # 2. google colab's GPU has been updated from K80 (compute_35) to T4 (compute_75).
        cuda9_gencode = (' -gencode arch=compute_60,code=sm_60'
                         ' -gencode arch=compute_70,code=sm_70')
        cuda10_gencode = ' -gencode arch=compute_75,code=sm_75'
        cuda11_gencode = ' -gencode arch=compute_80,code=sm_80'
        cuda9_ptx = ' -gencode arch=compute_70,code=compute_70'
        cuda10_ptx = ' -gencode arch=compute_75,code=compute_75'
        cuda11_ptx = ' -gencode arch=compute_80,code=compute_80'
        if cuda_major >= 11:
            gencode = cuda9_gencode + cuda10_gencode + cuda11_gencode + cuda11_ptx
        elif cuda_major >= 10:
            gencode = cuda9_gencode + cuda10_gencode + cuda10_ptx
        elif cuda_major >= 9:
            gencode = cuda9_gencode + cuda9_ptx
        else:
            raise CompileError(
                'CUDA version must be >=9.0, the current version is {}'.format(
                    cuda_major))

        extra_compile_args['nvcc'] = shlex.split(gencode) + [
            '-Xcompiler', '-fPIC'
        ]
    options = {
        'sources': sources,
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'libraries': libraries,
        'runtime_library_dirs': runtime_library_dirs,
        'extra_compile_args': extra_compile_args
    }

    return options


# credit: https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py#L55
def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', 'nvcc')
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    '''Customize the process for building the extension by chaning
    the options for compiling swig files and cu files.

    Ref: https://github.com/python/cpython/blob/master/Lib/distutils/command/build_ext.py
    '''

    def finalize_options(self):
        self.swig_cpp = True
        print('build temp', self.build_temp)
        print('build lib', self.build_lib)
        super(custom_build_ext, self).finalize_options()
        self.swig_opts = '-py3 -outdir {}/singa/'.format(self.build_lib).split()
        print('build temp', self.build_temp)
        print('build lib', self.build_lib)

    def build_extensions(self):
        options = prepare_extension_options()
        for key, val in options.items():
            singa_wrap.__dict__[key] = val
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


try:
    with io.open('README.md', encoding='utf-8') as f:
        long_description = '\n' + f.read()
except OSError:
    long_description = ''

classifiers = [
    # Trove classifiers
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    'License :: OSI Approved :: Apache Software License',
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
]
if sys.platform == 'darwin':
    classifiers.append('Operating System :: MacOS :: MacOS X')
elif sys.platform == 'linux':
    'Operating System :: POSIX :: Linux'
else:
    raise DistutilsSetupError('Building on Windows is not supported currently.')

keywords = 'deep learning, apache singa'
with_cuda, with_nccl, _, _ = parse_compile_options()
if with_cuda:
    classifiers.append('Environment :: GPU :: NVIDIA CUDA')
    cuda_version = os.environ.get('CUDA_VERSION')
    cudnn_version = os.environ.get('CUDNN_VERSION')
    keywords += ', cuda{}, cudnn{}'.format(cuda_version, cudnn_version)
    cuda_major = int(cuda_version.split('.')[0])
    cuda_minor = int(cuda_version.split('.')[1])
    # local label '+cuda10.2'. Ref: https://www.python.org/dev/peps/pep-0440/
    VERSION = VERSION + '+cuda{}.{}'.format(cuda_major, cuda_minor)
    if with_nccl:
        classifiers.append('Topic :: System :: Distributed Computing')
        keywords += ', distributed'
else:
    keywords += ', cpu-only'

singa_wrap = Extension('singa._singa_wrap', [])

setup(
    name=NAME,
    version=VERSION,
    description='A General Deep Learning System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Apache SINGA Community',
    author_email='dev@singa.apache.org',
    url='http://singa.apache.org',
    python_requires='>=3',
    install_requires=[
        'numpy >=1.16,<2.0',  #1.16
        'onnx==1.15',
        'deprecated',
        'pytest',
        'unittest-xml-reporting',
        'future',
        'pillow',
        'tqdm',
    ],
    include_package_data=True,
    license='Apache 2',
    classifiers=classifiers,
    keywords=keywords,
    packages=find_packages('python'),
    package_dir={'': 'python'},
    ext_modules=[singa_wrap],
    cmdclass={
        'build_ext': custom_build_ext,
        'audit': AuditCommand
    })
