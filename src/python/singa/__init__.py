# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================

'''
This script is the main entrance for user to run singa inside a model workspace

To use this script, user sudo install these dependencies: flask pillow and protobuf
'''

import sys, glob, os, random, shutil, time
from flask import Flask, request, redirect, url_for
import numpy as np
import ConfigParser
import urllib, traceback


from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
sys.path.append(os.getcwd())

__all__ = []
__version__ = 0.1
__date__ = '2016-07-20'
__updated__ = '2016-07-20'
__shortdesc__ = '''
welcome to singa
'''

app = Flask(__name__)
config = ConfigParser.RawConfigParser()
service = {}
data_path = "data_"
parameter_path = "parameter_"

debug = False

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    from . import device

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __shortdesc__
    program_license = '''%s

  Created by dbsystem group on %s.
  Copyright 2016 NUS School of Computing. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    global debug

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-p", "--port", dest="port", default=5000, help="the port to listen to, default is 5000")
        parser.add_argument("-param", "--parameter", dest="parameter",  help="the parameter file path to be loaded")
        parser.add_argument("-D", "--debug", dest="debug", action="store_true", help="whether need to debug")
        parser.add_argument("-R", "--reload", dest="reload_data", action="store_true", help="whether need to reload data")
        parser.add_argument("-C", "--cpu", dest="use_cpu", action="store_true", help="Using cpu or not, default is using gpu")
        parser.add_argument("-m", "--mode", dest="mode", choices=['train','test','serve'], default='serve', help="On Which mode (train,test,serve) to run singa")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)

        # Process arguments
        args = parser.parse_args()

        port = args.port
        parameter_file = args.parameter
        mode = args.mode
        need_reload = args.reload_data
        use_cpu = args.use_cpu
        debug = args.debug

        #prepare data files
        config.read('file.cfg')
        file_prepare(need_reload)


        import network as net
        model = net.create()

        #load parameter
        parameter_file=get_parameter(parameter_file)

        if parameter_file:
            print "load parameter file: %s" % parameter_file
            model.load(parameter_file)

        if use_cpu:
            raise CLIError("Currently cpu is not support!")
        else:
            print "runing with gpu"
            d = device.create_cuda_gpu()

        model.to_device(d)

        if mode == "serve":
            print "runing singa in serve mode, listen to  port: %s " % port
            global service
            from serve import Service
            service =Service(model,d)

            app.debug = debug
            app.run(host='0.0.0.0', port= port)
        elif mode == "train":
            print "runing singa in train mode"
            global trainer
            from train import Trainer
            trainer= Trainer(model,d)
            if not parameter_file:
                trainer.initialize()
            trainer.train()
        else:
            raise CLIError("Currently only serve mode is surpported!")
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        if debug:
            traceback.print_exc()
            raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + str(e) + "\n")
        sys.stderr.write(indent + "  for help use --help \n\n")
        return 2

def file_prepare(reload_data=False):
    '''
        download all files and generate data.py
    '''
    if not reload_data and os.path.exists("data_.py"):
        return

    print "download file"
    #clean data
    shutil.rmtree("data_.py",ignore_errors=True)
    shutil.rmtree("data_",ignore_errors=True)

    data_py=open("data_.py",'w')
    data_py.write("#%s" % "This file is Generated by SINGA, please don't edit\n\n")
    if config.has_section("data"):
        file_list = config.items("data")
        #download files
        for f in file_list:
            name,path=download_file(f[0],f[1],data_path)
            data_py.write("%s=\"%s\"\n" % (name,path))

    data_py.flush()
    data_py.close()

    if config.has_section("parameter"):
        parameter_list = config.items("parameter")
        for p in parameter_list:
            download_file(p[0],p[1],parameter_path)

def download_file(name,path,dest):
    '''
    download one file to dest
    '''
    if not os.path.exists(dest):
        os.makedirs(dest)
    if (path.startswith('http')):
        file_name = path.split('/')[-1]
        target = os.path.join(dest,file_name)
        urllib.urlretrieve(path,target)
    return name,target


def get_parameter(file_name=None):
    '''
    get the paticular file name or get the last parameter file
    '''
    if not os.path.exists(parameter_path):
        os.makedirs(parameter_path)
        return

    if file_name:
	return os.path.join(parameter_path,file_name)

    parameter_list = [ os.path.join(parameter_path,f) for f in os.listdir(parameter_path)]
    if len(parameter_list)==0:
        return
    parameter_list.sort()

    return parameter_list[-1]

@app.route("/")
def index():
    return "Hello SINGA User!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            response=service.serve(request)
        except Exception as e:
            return e
        return response
    return "error, should be post request"
