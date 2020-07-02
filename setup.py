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

from setuptools import find_packages, setup, Command, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError, DistutilsSetupError
from distutils import log as distutils_logger
from distutils.version import LooseVersion

import os
import io
import sys
import re
import shlex
import subprocess
import textwrap
import traceback
import pipes
import warnings
import shutil
from copy import deepcopy
from pathlib import Path

import conda.cli.python_api as conda
import numpy as np

NAME = 'singa'
# Update the version ID for new release following semantic version
VERSION='3.0.0.dev0'
CUDNN_VERSION = '7.6.5'
CUDA_VERSION = '10.2'

SINGA_PY = Path('python') #Path(__file__).parent
SINGA_SRC = Path('src')
SINGA_HDR = Path('include')

singa_extension = Extension(NAME, [])

def is_build_action():
    if len(sys.argv) <= 1:
        return False

    if sys.argv[1].startswith('build'):
        return True

    if sys.argv[1].startswith('bdist'):
        return True

    if sys.argv[1].startswith('install'):
        return True


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
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
        try:
            self.status('Removing previous builds…')
            shutil.rmtree(os.path.join(SINGA_PY, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        if 'dev' not in VERSION:
            self.status('Pushing git tags…')
            os.system('git tag v{0}'.format(VERSION))
            os.system('git push --tags')

        sys.exit()

def parse_compile_options():
    '''Read the environment variables to parse the compile options.

    Returns:
        a tuple of bool values as the indicators
    '''
    with_cuda = os.environ.get('SINGA_CUDA', False) 
    with_nccl = os.environ.get('SINGA_NCCL', False) 

    with_test = False
    if 'SINGA_TEST' in os.environ and os.environ['SINGA_TEST'].lower() == 'true':
        with_test = True
    with_debug = False
    if 'SINGA_DEBUG' in os.environ and os.environ['SINGA_DEBUG'].lower() == 'true':
        with_debug = True

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

    if with_nccl:
        config.append('#define ENABLE_DIST')
        config.append('#define USE_DIST')

    cpp_conf_path = SINGA_HDR/'singa/singa_config.h'
    print('Writing configs to {}'.format(cpp_conf_path))
    with cpp_conf_path.open('w') as fd:
        for line in config:
            fd.write(line + '\n')

    swig_conf_path = SINGA_SRC/'api/config.i'
    with swig_conf_path.open('w') as fd:
        for line in config:
            fd.write(line + ' 1 \n')
        fd.write('#define CUDNN_VERSION "{}"\n'.format(CUDNN_VERSION))
        versions = [int(x) for x in VERSION.split('.')[:3]]
        fd.write('#define SINGA_MAJOR_VERSION {}\n'.format(versions[0]))
        fd.write('#define SINGA_MINOR_VERSION {}\n'.format(versions[1]))
        fd.write('#define SINGA_PATCH_VERSION {}\n'.format(versions[2]))
        fd.write('#define SINGA_VERSION "{}"\n'.format(VERSION))


def install_dep_lib(prefix, with_cuda, with_nccl):
    '''Install dependent libs via conda.

    Args:
        prefix(str): location to put the libs
        with_cuda(bool): indicator for cudnn and cuda lib
        with_nccl(bool): indicator for nccl lib

    Returns:
        a list of link names of dependent libs; 
        and a list of extra libs with indirect dependency
    '''
    # stdout, stderr, _ = conda.run_command(conda.Commands.INFO)
    deps = ['glog=0.3.5', 'openblas=0.3.9', 'dnnl=1.1', 'protobuf=3.9.2']
    link_libs = [x.split('=')[0] for x in deps]

    if with_nccl:
        deps.append('nccl=2.6.4.1')
        deps.append('mpich=3.3.2')
        link_libs +=['nccl', 'mpi', 'mpicxx']

    if with_cuda:
        deps.append('cudnn=7.6.5=cuda10.2_0')
        link_libs += ['cudnn', 'cudart', 'curand', 'cnmem']


    for dep in deps:
        print('-------------------------------')
        print('Install  {} '.format(dep))
        print('-------------------------------')
        stdout, stderr, _ = conda.run_command(conda.Commands.INSTALL, 
            '--prefix', str(prefix), '-c', 'conda-forge', '-c', 'nusdbsystem', dep)
        print(stderr)
        print(stdout)
    conda.run_command(conda.Commands.INSTALL, '-c', 'conda-forge', 'swig=3.0.12')

    extra_libs = ['gflags', 'gomp', 'gfortran', 'quadmath']
    return link_libs, extra_libs


def test_compile(build_ext, name, code, libraries=None, include_dirs=None, library_dirs=None,
                 macros=None, extra_compile_preargs=None, extra_link_preargs=None):
    test_compile_dir = os.path.join(build_ext.build_temp, 'test_compile')
    if not os.path.exists(test_compile_dir):
        os.makedirs(test_compile_dir)

    source_file = os.path.join(test_compile_dir, '%s.cc' % name)
    with open(source_file, 'w') as f:
        f.write(code)

    compiler = build_ext.compiler
    [object_file] = compiler.object_filenames([source_file])
    shared_object_file = compiler.shared_object_filename(
        name, output_dir=test_compile_dir)

    compiler.compile([source_file], extra_preargs=extra_compile_preargs,
                     include_dirs=include_dirs, macros=macros)
    compiler.link_shared_object(
        [object_file], shared_object_file, libraries=libraries, library_dirs=library_dirs,
        extra_preargs=extra_link_preargs)

    return shared_object_file

def get_cpp_flags(build_ext=None):
    last_err = None
    default_flags = ['-std=c++11', '-fPIC', '-g', '-O2', '-Wall', '-pthread']
    # avx_flags = [ '-mavx'] #'-mf16c',
    flags_to_try = []
    if sys.platform == 'darwin':
        # Darwin most likely will have Clang, which has libc++.
        flags_to_try = [default_flags + ['-stdlib=libc++'],
                        default_flags]
    else:
        flags_to_try = [default_flags,
                        default_flags + ['-stdlib=libc++']]
    return flags_to_try[0]
    for cpp_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cpp_flags', extra_compile_preargs=cpp_flags,
                         code=textwrap.dedent('''\
                    #include <unordered_map>
                    void test() {
                    }
                    '''))

            return cpp_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ compilation flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ compilation flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_link_flags(build_ext=None):
    last_err = None
    libtool_flags = []
    ld_flags = []
    flags_to_try = []
    if sys.platform == 'darwin':
        flags_to_try = [libtool_flags, ld_flags]
    else:
        flags_to_try = [ld_flags, libtool_flags]
    return flags_to_try[0]
    for link_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_link_flags', extra_link_preargs=link_flags,
                         code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            return link_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ link flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ link flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def test_cuda_libs(build_ext, prefix, cpp_flags):
    cuda_include_dirs = [prefix/'include']
    cuda_lib_dirs = [prefix/'lib', prefix/'lib64']
    cuda_libs = ['cudart']

    try:
        test_compile(build_ext, 'test_cuda', libraries=cuda_libs,
                     include_dirs=cuda_include_dirs,
                     library_dirs=cuda_lib_dirs,
                     extra_compile_preargs=cpp_flags,
                     code=textwrap.dedent('''\
            #include <cuda_runtime.h>
            void test() {
                cudaSetDevice(0);
            }
            '''))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'CUDA library was not found (see error above).\n'
            'Please specify correct CUDA location with the CUDA_HOME '
            'HOROVOD_CUDA_HOME - path where CUDA include and lib directories can be found\n')

    return cuda_include_dirs, cuda_lib_dirs


def generate_proto_files(prefix):
    print('----------------------')
    print('Generating proto files')
    print('----------------------')
    proto_src = SINGA_SRC/'proto'
    cmd = "{}/bin/protoc --proto_path={} --cpp_out={} {}".format(prefix, proto_src, proto_src,  proto_src/'core.proto')
    subprocess.run(cmd, shell=True, check=True)

    proto_hdr_dir = SINGA_HDR/'singa/proto'
    proto_hdr_file = proto_hdr_dir/'core.pb.h'
    if proto_hdr_dir.exists():
        if proto_hdr_file.exists():
            proto_hdr_file.unlink()
    else:
        proto_hdr_dir.mkdir()

    shutil.copyfile(Path(proto_src/'core.pb.h'), proto_hdr_file)
    return proto_hdr_file, proto_src/'core.pb.cc'

def compile_cuda_kernel():
    pass

def path_to_str(path_list):
    return [str(x) if not isinstance(x, str) else x for x in path_list]

def prepare_extension_options():
    with_cuda, with_nccl, with_test, with_debug = parse_compile_options()
    prefix = Path(SINGA_PY/'singa/thirdparty')
    prefix.mkdir(exist_ok=True)
    link_libs, extra_libs = install_dep_lib(prefix, with_cuda, with_nccl)
    print(link_libs, extra_libs)
    package_data = ['thirdparty/lib/lib{}.so.[0-9]'.format(x) for x in link_libs + extra_libs]
    generate_singa_config(with_cuda, with_nccl)
    generate_proto_files(prefix)

    sources = path_to_str([*list((SINGA_SRC/'core').rglob('*.cc')), 
                           *list((SINGA_SRC/'model/operation').glob('*.cc')), 
                           *list((SINGA_SRC/'utils').glob('*.cc')),
                            SINGA_SRC/'proto/core.pb.cc',
                            SINGA_SRC/'api/singa.i'])
    include_dirs = path_to_str([SINGA_HDR, SINGA_HDR/'singa/proto', np.get_include(), prefix/'include'])
    library_dirs = path_to_str([prefix/'lib'])
    libraries = link_libs
    runtime_library_dirs = path_to_str(['.', prefix/'lib'])

    extra_compile_args = get_cpp_flags()
    extra_link_args = get_link_flags()
    options = {'sources':sources, 
               'include_dirs':include_dirs, 
               'library_dirs':library_dirs,
               'libraries':libraries,
               'runtime_library_dirs':runtime_library_dirs,
               'extra_compile_args':extra_compile_args,
               'extra_link_args':extra_link_args}

    classifiers = [
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]

    if sys.platform == 'darwin':
        classifiers.append('Operating System :: MacOS :: MacOS X')
    elif sys.platform == 'linux':
        'Operating System :: POSIX :: Linux'
    else:
        raise DistutilsSetupError('Building on Windows is not supported currently.')

    if with_cuda:
        classifiers.append('Environment :: GPU :: NVIDIA CUDA')
        if with_nccl:
            classifiers.append('Topic :: System :: Distributed Computing')
    return options, package_data, classifiers

options, package_data, classifiers = prepare_extension_options()
singa_wrap = Extension('singa._singa_wrap', **options)


class custom_build_ext (build_ext):
    def finalize_options(self):
        self.swig_cpp = True
        self.swig_opts = '-outdir {}'.format(SINGA_PY/'singa')
        self.rpath = '$ORIGIN/thirdparty/lib'
        super(custom_build_ext, self).finalize_options()

try:
    with io.open('README.md', encoding='utf-8') as f:
        long_description = '\n' + f.read()
except OSError:
    long_description = ''

setup(
    name=NAME,
    version=VERSION,
    description='A General Deep Learning System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Apache SINGA Community',
    author_email='dev@singa.apache.org',
    url='http://singa.apache.org',
    install_requires=[
        'numpy >=1.16,<2.0', #1.16
        'onnx==1.6',
        'deprecated',
        'unittest-xml-reporting',
        'future',
        'pillow',
        'tqdm',
        ],
    include_package_data=True,
    license='Apache 2',
    classifiers=classifiers,
    keywords='deep learning singa apache',
    #List additional groups of dependencies here (e.g. development
    #dependencies). You can install these using the following syntax,
    #for example:
    #$ pip install -e .[dev,test]
    #extras_require={
    #   'dev': ['check-manifest'],
    #   'test': ['coverage'],
    #},

    #If there are data files included in your packages that need to be
    #installed, specify them here.  If using Python 2.6 or less, then these
    #have to be included in MANIFEST.in as well.
    packages=find_packages('python'),
    package_dir = {'':'python'},
    package_data={
        'singa': package_data,
    },
    ext_modules=[singa_wrap],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
        'build_ext': custom_build_ext
    },
)