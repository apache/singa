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
from builtins import str
from builtins import object

from multiprocessing import Process, Queue
from flask import Flask,request, send_from_directory, jsonify
from flask_cors import cross_origin
import os, traceback
import time
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict, MultiDict
import pickle
import uuid

class MsgType(object):
   def __init__(self, name):
       self.name = name
   def __str__(self):
       return self.name
   def __repr__(self):
       return "<Msg: %s>" % self
   def equal(self,target):
       return str(self) == str(target)

   def is_info(self):
       return self.name.startswith('kInfo')
   def is_command(self):
       return self.name.startswith('kCommand')
   def is_status(self):
       return self.name.startswith('kStatus')
   def is_request(self):
       return self.name.startswith('kRequest')
   def is_response(self):
       return self.name.startswith('kResponse')

   @staticmethod
   def parse(name):
       return getattr(MsgType,str(name))
   @staticmethod
   def get_command(name):
       if name=='stop':
           return MsgType.kCommandStop
       if name=='pause':
           return MsgType.kCommandPause
       if name=='resume':
           return MsgType.kCommandResume
       return MsgType.kCommand

types =  ['kInfo','kInfoMetric',
           'kCommand','kCommandStop','kCommandPause','kCommandResume',
           'kStatus','kStatusRunning','kStatusPaused','kStatusError',
           'kRequest','kResponse']

for t in types:
    setattr(MsgType,t,MsgType(t))

#####   NOTE the server currently only can handle request sequentially

app = Flask(__name__)
top_k_=5

class Agent(object):

    def __init__(self,port):
        info_queue = Queue()
        command_queue = Queue()
        self.p = Process(target=start, args=(port, info_queue,command_queue))
        self.p.start()
        self.info_queue=info_queue
        self.command_queue=command_queue
        return

    def pull(self):
        if not self.command_queue.empty():
            msg,data=self.command_queue.get()
            if msg.is_request():
                data = pickle.loads(data)
            return msg,data
        return None,None

    def push(self,msg,value):
        self.info_queue.put((msg,value))
        return

    def stop(self):
        #sleep a while, wait for http response finished
        time.sleep(1)
        self.p.terminate()

def start(port,info_queue,command_queue):
    global info_queue_, command_queue_, data_
    info_queue_=info_queue
    command_queue_=command_queue
    data_ = []
    app.run(host='0.0.0.0', port=port)
    return

def getDataFromInfoQueue(need_return=False):
    global info_queue_, data_
    if not need_return:
        while not info_queue_.empty():
            msg,d = info_queue_.get()
            data_.append(d)
    else:
        while True: # loop until get answer
            while not info_queue_.empty():
                msg,d = info_queue_.get()
                if msg.is_info():
                    data_.append(d)
                else:
                    return msg,d
            time.sleep(0.01)

@app.route("/")
@cross_origin()
def index():
    try:
        req=send_from_directory(os.getcwd(),"index.html", mimetype='text/html')
    except Exception:
        traceback.print_exc()
        return "error"
    except SystemExit:
        traceback.print_exc()
        return "error"
    return req

# support two operations for user to monitor the training status
@app.route('/getAllData')
@cross_origin()
def getAllData():
    global data_
    try:
        getDataFromInfoQueue()
    except Exception:
        traceback.print_exc()
        return failure("Internal Error - Triggered Exception")
    except SystemExit:
        traceback.print_exc()
        return failure("Internal Error - Triggered SystemExit")
    return success(data_)


@app.route('/getTopKData')
@cross_origin()
def getTopKData():
    global data_
    try:
        k = int(request.args.get("k", top_k_))
    except Exception:
        traceback.print_exc()
        return failure("k should be integer - Triggered Exception")
    except SystemExit:
        traceback.print_exc()
        return failure("k should be integer - Triggered SystemExit")
    try:
        getDataFromInfoQueue()
    except Exception:
        traceback.print_exc()
        return failure("Internal Error - Triggered Exception")
    except SystemExit:
        traceback.print_exc()
        return failure("Internal Error - Triggered SystemExit")
    return success(data_[-k:])

@app.route("/api", methods=['POST'])
@cross_origin()
def api():
    global info_queue_,command_queue_
    try:
        files=transformFile(request.files)
        values = CombinedMultiDict([request.args,request.form,files])
        req_str = pickle.dumps(values)
        command_queue_.put((MsgType.kRequest,req_str))
        msg,response=getDataFromInfoQueue(True)
        deleteFiles(files)
        return response
    except Exception:
        traceback.print_exc()
        return failure("Internal Error - Triggered Exception")
    except SystemExit:
        traceback.print_exc()
        return failure("Internal Error - Triggered SystemExit")

@app.route("/command/<name>", methods=['GET','POST'])
@cross_origin()
def command(name):
    global info_queue_,command_queue_
    try:
        command=MsgType.get_command(name)
        command_queue_.put((command,""))
        msg,response=getDataFromInfoQueue(True)
        return response
    except Exception:
        traceback.print_exc()
        return failure("Internal Error - Triggered Exception")
    except SystemExit:
        traceback.print_exc()
        return failure("Internal Error - Triggered SystemExit")

def success(data=""):
    '''return success status in json format'''
    res = dict(result="success", data=data)
    return jsonify(res)
def failure(message):
    '''return failure status in json format'''
    res = dict(result="message", message=message)
    return jsonify(res)


def transformFile(files):
    result= MultiDict([])
    for f in files:
        file = files[f]
        unique_filename = str(uuid.uuid4())+secure_filename(file.filename)
        filepath=os.path.join(os.getcwd(), unique_filename)
        file.save(filepath)
        result.add(f, filepath)
    return result


def deleteFiles(files):
    for f in files:
        filepath = files[f]
        os.remove(filepath)
    return
