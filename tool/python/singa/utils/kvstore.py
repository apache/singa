'''
Created on Jan 8, 2016

@author: aaron
'''
import struct, os 

INT_LEN=8

class FileStore():
    '''
    kv file store
    '''
    def open(self,src_path,mode):
        if mode == "create":
            self._file = open(src_path,"wb") 
        if mode == "append":
            self._file = open(src_path,"ab")
        if mode == "read":
            self._file = open(src_path,"rb")
        return self

    def close(self):
        self._file.close()
        return

    def read(self):
        keyLen=b2i(self._file.read(INT_LEN))
        key=str(self._file.read(keyLen))
        valueLen=b2i(self._file.read(INT_LEN))
        value=str(self._file.read(valueLen))
        return key,value

    def seekToFirst(self):
        self._file.seek(0)
        return
    
    #Don't do this
    def seek(self,offset):
        self._file.seek(offset)
        return

    def write(self,key,value):
        key_len = len(key)
        value_len = len(value)
        self._file.write(i2b(key_len)+key+i2b(value_len)+value)
        return

    def flush(self):
        self._file.flush()
        return

    def __init__(self ):

        return
#integer to binary Q means long long, 8 bytes
def i2b(i):
    return struct.pack("<Q",i)
#binary to integer
def b2i(b):
    return struct.unpack("<Q",b)[0]

if __name__=='__main__':
    store=FileStore()
    store.open("test","create")
    store.write("Hello","world!")
    store.flush()
    store.close()
    
    store.open("test","read")
    key,value=store.read()
    print key, value
    store.close()
    
    store.open("test","append")
    store.write("Foo","Bar")
    store.flush()
    store.close()
 
    store.open("test","read")
    key,value=store.read()
    print key,value
    key,value=store.read()
    print key,value
    store.seekToFirst()
    key,value=store.read()
    print key,value
    store.close()
    
    os.remove("test")
