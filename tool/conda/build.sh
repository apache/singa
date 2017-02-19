export CMAKE_PREFIX_PATH=$PREFIX
export export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

echo $PREFIX

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..
make
make install
