#!/usr/bin/env bash

function install_cmake()
{
	if [ ! -e "cmake-3.2.1.tar.gz" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/cmake-3.2.1.tar.gz;
	fi

	rm -rf cmake-3.2.1;
	tar zxvf cmake-3.2.1.tar.gz && cd cmake-3.2.1;

	if [ $# == 1 ]
		then
			echo "install cmake in $1";
			./configure --prefix=$1;
			make && make install;
		elif [ $# == 0 ]
		then
			echo "install cmake in default path";
			./configure;
			make && sudo make install;
		else
			echo "wrong commands";
	fi

	if [ $? -ne 0 ]
	then
		cd ..;
		return -1;
	fi

	PATH=$1/bin:$PATH;
	echo $PATH;
	export PATH=$1/bin:$PATH;
	cd ..;
	return 0;
}

function install_czmq()
{
	if [ ! -e "czmq-3.0.0-rc1.tar.gz" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/czmq-3.0.0-rc1.tar.gz;
	fi
	rm -rf czmq-3.0.0;
	tar zxvf czmq-3.0.0-rc1.tar.gz && cd czmq-3.0.0;
	
	if [ $# == 2 ]
	then
		if [ $1 == "null" ]
		then
			echo "install czmq in default path. libzmq path is $2";
			./configure --with-libzmq=$2;
			make && sudo make install;
		else
			echo "install czmq in $1. libzmq path is $2";
			./configure --prefix=$1 --with-libzmq=$2;
			make && make install;
		fi
	elif [ $# == 1 ]
	then
		if [ $1 == "null" ]
		then
			echo "install czmq in default path.";
			./configure;
			make && sudo make install;
		else
			echo "install czmq in $1";
			./configure --prefix=$1;
			make && make install;
		fi
	else
		echo "ERROR: wrong command.";
		return -1;
	fi

	if [ $? -ne 0 ]
	then
		cd ..;
		return -1;
	fi
	cd ..;
	return 0;
}

function install_glog()
{
	if [ ! -e "glog-0.3.3.tar.gz" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/glog-0.3.3.tar.gz;
	fi
	
	rm -rf glog-0.3.3;
	tar zxvf glog-0.3.3.tar.gz && cd glog-0.3.3;

	if [ $# == 1 ]
		then
			echo "install glog in $1";
			./configure --prefix=$1;
			make && make install;
		elif [ $# == 0 ]
		then
			echo "install glog in default path";
			./configure;
			make && sudo make install;
		else
			echo "wrong commands";
	fi

	if [ $? -ne 0 ]
	then
		cd ..;
		return -1;
	fi
	cd ..;
	return 0;
}

function install_lmdb()
{
	if [ ! -e "lmdb-0.9.10.tar.gz" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/lmdb-0.9.10.tar.gz;
	fi

	rm -rf mdb-mdb;
	tar zxvf lmdb-0.9.10.tar.gz && cd mdb-mdb/libraries/liblmdb;

	if [ $# == 1 ]
		then
			echo "install lmdb in $1";
			sed -i "26s#^.*#prefix=$1#g" Makefile;
			make && make install;
		elif [ $# == 0 ]
		then
			echo "install lmdb in default path";
			make && sudo make install;
		else
			echo "wrong commands";
	fi

	if [ $? -ne 0 ]
	then
		cd ../../..;
		return -1;
	fi
	cd ../../..;
	return 0;
}

function install_openblas()
{
	if [ ! -e "OpenBLAS.zip" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/OpenBLAS.zip;
	fi

	rm -rf OpenBLAS-develop;
	unzip OpenBLAS.zip && cd OpenBLAS-develop;

	make;
	if [ $? -ne 0 ]
	then
		cd ..;
		return -1;
	fi
	if [ $# == 1 ]
		then
			echo "install OpenBLAS in $1";
			make PREFIX=$1 install;
			if [ $? -ne 0 ]
			then
				cd ..;
				return -1;
			fi
		elif [ $# == 0 ]
		then
			echo "install OpenBLAS in default path" 
			sudo make install;
			if [ $? -ne 0 ]
			then
				cd ..;
				return -1;
			fi
		else
			echo "wrong commands";
	fi
	cd ..;

	return 0;
}

function install_opencv()
{
	if [ ! -e "opencv-2.4.10.zip" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/opencv-2.4.10.zip;
	fi

	rm -rf opencv-2.4.10;
	unzip opencv-2.4.10.zip && cd opencv-2.4.10;
	
	if [ $# == 1 ]
		then
			echo "install opencv in $1";
			cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=$1;
			make && make install;
		elif [ $# == 0 ]
		then
			echo "install opencv in default path";
			cmake -DCMAKE_BUILD_TYPE=RELEASE;
			make && sudo make install;
		else
			echo "wrong commands";
	fi

	if [ $? -ne 0 ]
	then
		cd ..;
		return -1;
	fi
	cd ..;	
	return 0;
}

function install_protobuf()
{
	if [ ! -e "protobuf-2.6.0.tar.gz" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/protobuf-2.6.0.tar.gz;
	fi

	rm -rf protobuf-2.6.0;
	tar zxvf protobuf-2.6.0.tar.gz && cd protobuf-2.6.0;

	if [ $# == 1 ]
		then
			echo "install protobuf in $1";
			./configure --prefix=$1;
			make && make install;
			cd python;
			python setup.py build;
			python setup.py install --prefix=$1;
			cd ..;
		elif [ $# == 0 ]
		then
			echo "install protobuf in default path";
			./configure;
			make && sudo make install;
			cd python;
			python setup.py build;
			sudo python setup.py install;
			cd ..;
		else
			echo "wrong commands";
	fi

	if [ $? -ne 0 ]
	then
		cd ..;
		return -1;
	fi
	cd ..;
	return 0;
}

function install_zeromq()
{
	if [ ! -e "zeromq-3.2.2.tar.gz" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/zeromq-3.2.2.tar.gz;
	fi

	rm -rf zeromq-3.2.2;
	tar zxvf zeromq-3.2.2.tar.gz && cd zeromq-3.2.2;

	if [ $# == 1 ]
		then
			echo "install zeromq in $1";
			./configure --prefix=$1;
			make && make install;
		elif [ $# == 0 ]
		then
			echo "install zeromq in default path";
			./configure;
			make && sudo make install;
		else
			echo "wrong commands";
	fi
	
	if [ $? -ne 0 ]
	then
		cd ..;
		return -1;
	fi
	cd ..;
	return 0;
}

function install_zookeeper()
{
	if [ ! -e "zookeeper-3.4.6.tar.gz" ]
	then
		wget http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/thirdparty/zookeeper-3.4.6.tar.gz;
	fi

	rm -rf zookeeper-3.4.6;
	tar zxvf zookeeper-3.4.6.tar.gz;
	cd zookeeper-3.4.6/src/c;

	if [ $# == 1 ]
		then
			echo "install zookeeper in $1";
			./configure --prefix=$1;
			make && make install;
		elif [ $# == 0 ]
		then
			echo "install zookeeper in default path";
			./configure;
			make && sudo make install;
		else 
			echo "wrong commands";
	fi

	if [ $? -ne 0 ]
	then
		cd ../../..;
		return -1;
	fi

	cd ../../..;
	return 0;
}

BIN=`dirname "${BASH_SOURCE-$0}"`
BIN=`cd "$BIN">/dev/null; pwd`
BASE=`cd "$BIN/..">/dev/null; pwd`

cd $BIN

while [ $# != 0 ]
do
	case $1 in
	"cmake")
		echo "install cmake";
		if [[ $2 == */* ]];then
			install_cmake $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during cmake installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_cmake;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during cmake installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	"czmq")
		echo "install czmq";
		if [ $2 == "-f" ]
		then
			if [[ $4 == */* ]]
			then
				install_czmq $4 $3;
			else
				install_czmq null $3;
			fi
			if [ $? -ne 0 ] 
			then
				echo "ERROR during czmq installation" ;
				exit;
			fi
			shift
			shift
			shift
			shift
		elif [ $3 == "-f" ]
		then
			install_czmq $2 $4;
			if [ $? -ne 0 ] 
			then
				echo "ERROR during czmq installation" ;
				exit;
			fi
			shift
			shift
			shift
			shift
		elif [[ $2 == */* ]]
		then
			install_czmq $2;
			if [ $? -ne 0 ] 
			then
				echo "ERROR during czmq installation" ;
				exit;
			fi
			shift
			shift
		else
			install_czmq null;
			if [ $? -ne 0 ] 
			then
			    echo "ERROR during czmq installation" ;
			    exit;
			fi  
			shift
		fi
		;;
	"glog")
		echo "install glog";
		if [[ $2 == */* ]];then
			install_glog $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during glog installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_glog;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during glog installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	"lmdb")
		echo "install lmdb";
		if [[ $2 == */* ]];then
			install_lmdb $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during lmdb installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_lmdb;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during lmdb installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	"OpenBLAS")
		echo "install OpenBLAS";
		if [[ $2 == */* ]];then
			install_openblas $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during openblas installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_openblas;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during openblas installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	"opencv")
		echo "install opencv";
		if [[ $2 == */* ]];then
			install_opencv $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during opencv installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_opencv;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during opencv installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	 "protobuf")
		echo "install protobuf";
		if [[ $2 == */* ]];then
			install_protobuf $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during protobuf installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_protobuf;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during protobuf installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	"zeromq")
		echo "install zeromq";
		if [[ $2 == */* ]];then
			install_zeromq $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during zeromq installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_zeromq;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during zeromq installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	"zookeeper")
		echo "install zookeeper";
		if [[ $2 == */* ]];then
			install_zookeeper $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during zookeeper installation" ;
		        exit;
		    fi  
			shift
			shift
		else
			install_zookeeper;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during zookeeper installation" ;
		        exit;
		    fi  
			shift
		fi
		;;
	"all")
		echo "install all dependencies";
		if [[ $2 == */* ]];then
			install_cmake $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during cmake installation" ;
		        exit;
		    fi  
			install_zeromq $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during zeromq installation" ;
		        exit;
		    fi  
			install_czmq $2 $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during czmq installation" ;
		        exit;
		    fi  
			install_glog $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during glog installation" ;
		        exit;
		    fi  
			install_lmdb $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during lmdb installation" ;
		        exit;
		    fi  
			install_openblas $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during openblas installation" ;
		        exit;
		    fi  
			install_opencv $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during opencv installation" ;
		        exit;
		    fi  
			install_protobuf $2;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during protobuf installation" ;
		        exit;
		    fi  
			install_zookeeper $2;
			if [ $? -ne 0 ]
			then
				echo "ERROR during zookeeper installation" ;
				exit;
			fi
			shift
			shift
		else
			install_cmake;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during cmake installation" ;
		        exit;
		    fi  
			install_zeromq;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during zeromq installation" ;
		        exit;
		    fi  
			install_czmq null;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during czmq installation" ;
		        exit;
		    fi  
			install_glog;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during glog installation" ;
		        exit;
		    fi  
			install_lmdb;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during lmdb installation" ;
		        exit;
		    fi  
			install_openblas;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during openblas installation" ;
		        exit;
		    fi  
			install_opencv;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during opencv installation" ;
		        exit;
		    fi  
			install_protobuf;
		    if [ $? -ne 0 ] 
		    then
		        echo "ERROR during protobuf installation" ;
		        exit;
		    fi  
			install_zookeeper;
			if [ $? -ne 0 ]
			then
				echo "ERROR during zookeeper installation" ;
				exit;
			fi
			shift
		fi
		;;
	*)
		echo "USAGE: ./install.sh [MISSING_LIBRARY_NAME1] [YOUR_INSTALL_PATH1] [MISSING_LIBRARY_NAME2] [YOUR_INSTALL_PATH2] ...";
		echo " MISSING_LIBRARY_NAME can be:	"
		echo "	cmake"
		echo "	czmq"
		echo "	glog"
		echo "	lmdb"
		echo "	OpenBLAS"
		echo "	opencv"
		echo "	protobuf"
		echo "	zeromq"
		echo "	zookeeper"
		echo " To install all dependencies, you can run:	"
		echo "	./install.sh all"
		exit;
	esac	
done
