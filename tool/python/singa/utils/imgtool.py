'''
Created on Jan 8, 2016
@author: aaron
'''
from PIL import Image
import sys, glob, os, random, shutil, time, struct
from . import kvstore

sys.path.append(os.path.join(os.path.dirname(__file__), '../../pb2'))
from common_pb2 import RecordProto

#bytearray to image object
def toImg(byteArray,size):
    img = Image.new("RGB",size)
    pix = img.load()
    area = size[0]*size[1]
    red = byteArray[:area]
    green = byteArray[area:area*2]
    blue = byteArray[area*2:]
    index=0
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            img.putpixel((x,y), (red[index],green[index],blue[index]))     
            index+=1
    return img

# image object to bytearray
def toBin(im,size):
    red = []
    green = []
    blue = []
    pix = im.load()
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            pixel = pix[x,y]
            red.append(pixel[0])
            green.append(pixel[1])
            blue.append(pixel[2])         
    fileByteArray = bytearray(red+green+blue)
    return fileByteArray

def resize_to_center(im,size):
    oldSize = im.size
    #bigest center cube
    data=(0,0,0,0)
    if oldSize[0] < oldSize[1]:
        data= (0,(oldSize[1]-oldSize[0])/2,oldSize[0],(oldSize[1]+oldSize[0])/2)
    else :
        data= ((oldSize[0]-oldSize[1])/2,0,(oldSize[0]+oldSize[1])/2,oldSize[1])
    newIm = im.transform(size,Image.EXTENT,data)
    return newIm
#transfer, resize img. only deal with .jpg file
def transform_img(
            input_folder,
            output_folder, 
            size         
                     ):
    print "Transfer images begin at:"+time.strftime('%X %x %Z')

    #if output_folder exists, empty it, otherwise create a dir
    try:
        os.stat(output_folder)
        for root, dirs, files in os.walk(output_folder):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    except:
        os.makedirs(output_folder)

    count=0
    for root, dirs, files in os.walk(input_folder):
        for d in dirs:
            print "find dir:", d
            os.makedirs(os.path.join(output_folder,d))
            for infile in glob.glob(os.path.join(input_folder,d,"*.jpg")):
                fileName = os.path.split(infile)[-1]
                name,ext = os.path.splitext(fileName) 
                im = Image.open(infile).convert("RGB")
                newIm=resize_to_center(im,size)
                newIm.save(os.path.join(output_folder,d,name+".center.jpg"), "JPEG")
                count+=1

    print "transfer end at:"+time.strftime('%X %x %Z')
    print "total file number: ", count

    return count

               

def generate_bin_data(
                input_folder,
                output_folder,
                size ,     
                train_num,
                test_num,
                validate_num,
                meta_file_name="meta.txt",
                train_bin_file_name="train.bin",
                train_label_bin_file_name="train.label.bin",
                test_bin_file_name="test.bin",
                test_label_bin_file_name="test.label.bin",
                validate_bin_file_name="validate.bin",
                validate_label_bin_file_name="validate.label.bin",
                mean_bin_file_name="mean.bin",
                label_bin_file_name="label.bin",

                      ):
    try:
        os.stat(output_folder)
    except:
        os.makedirs(output_folder)

    print "Generate bin start at: "+time.strftime('%X %x %Z') 
    meta_file = open(os.path.join(output_folder,meta_file_name), "w")

    fileList=[]
    labelList= []
    label=0 #label begin from 1

    #get all img file, the folder name is the label name
    for d in os.listdir(input_folder):    
        if os.path.isdir(os.path.join(input_folder,d)):
            labelList.append((label,d))
            for f in glob.glob(os.path.join(input_folder,d,"*.jpg")):
                fileList.append((label,f))
            label += 1

    # disorder all the files
    random.shuffle(fileList)

    total = len(fileList)
    print total,train_num,test_num,validate_num
    assert total >= train_num+test_num+validate_num

    train_file = open(os.path.join(output_folder,train_bin_file_name),"wb")    
    train_label_file = open(os.path.join(output_folder,train_label_bin_file_name),"wb")    
    validate_file = open(os.path.join(output_folder,validate_bin_file_name),"wb")    
    validate_label_file = open(os.path.join(output_folder,validate_label_bin_file_name),"wb")    
    test_file = open(os.path.join(output_folder,test_bin_file_name),"wb")    
    test_label_file = open(os.path.join(output_folder,test_label_bin_file_name),"wb")    
    mean_file = open(os.path.join(output_folder,mean_bin_file_name),"wb")    
 
    count=0
    trainCount=0
    validateCount=0
    testCount=0

    # the expected image binary length
    binaryLength=3*size[0]*size[1] 

    meanData=[]
    for i in range(0,binaryLength):
        meanData.append(0.0)

    #calculate mean
    for (label,f) in fileList:    

        count+=1
        im =Image.open(f)
        #the image size should be equal
        assert im.size==size
        binaryPixel=toBin(im,size)
        if count <= train_num :
            trainCount+=1
            train_file.write(binaryPixel) 
            train_label_file.write(kvstore.i2b(label))
            #only caculate train data's mean value
            for i in range(binaryLength):
                meanData[i]+=binaryPixel[i]
        elif count <= train_num+validate_num :
            validateCount+=1
            validate_label_file.write(kvstore.i2b(label))
            validate_file.write(binaryPixel) 
        elif count <= train_num+validate_num+test_num:
            testCount+=1
            test_label_file.write(kvstore.i2b(label))
            test_file.write(binaryPixel) 
	else:
            break

    for i in range(binaryLength):
        meanData[i]/=trainCount

    meanBinary=struct.pack("%sf" % binaryLength, *meanData)

    mean_file.write(meanBinary)
    mean_file.flush()
    mean_file.close()
     
    train_file.flush()
    train_file.close()
    validate_file.flush()
    validate_file.close()
    test_file.flush()
    test_file.close()

    meta_file.write("image size: "+str(size[0])+"*"+str(size[1])+"\n")    
    meta_file.write("total file num: "+str(count)+"\n")    
    meta_file.write("train file num: "+str(trainCount)+"\n")
    meta_file.write("validate file num: "+str(validateCount)+"\n")
    meta_file.write("test file num: "+str(testCount)+"\n")
    meta_file.write("label list:[\n")

    for item in labelList:
        meta_file.write("("+str(item[0])+",\""+item[1]+"\"),\n")
    meta_file.write("]")
    meta_file.flush()
    meta_file.close()

    print "end at: "+time.strftime('%X %x %Z')    

    return labelList


def generate_kvrecord_data(
                input_folder,
                output_folder,
                size ,     
                train_num,
                test_num,
                validate_num,
                meta_file_name="meta.txt",
                train_bin_file_name="train.bin",
                test_bin_file_name="test.bin",
                validate_bin_file_name="validate.bin",
                mean_bin_file_name="mean.bin",

                      ):
    try:
        os.stat(output_folder)
    except:
        os.makedirs(output_folder)

    print "Generate kvrecord start at: "+time.strftime('%X %x %Z') 
    meta_file = open(os.path.join(output_folder,meta_file_name), "w")

    fileList=[]
    labelList= []
    label=0 #label begin from 1

    #get all img file, the folder name is the label name
    for d in os.listdir(input_folder):    
        if os.path.isdir(os.path.join(input_folder,d)):
            labelList.append((label,d))
            for f in glob.glob(os.path.join(input_folder,d,"*.jpg")):
                fileList.append((label,f))
            label += 1

    # disorder all the files
    random.shuffle(fileList)

    total = len(fileList)
    print total,train_num,test_num,validate_num
    assert total >= train_num+test_num+validate_num


    trainStore = kvstore.FileStore()
    trainStore.open(os.path.join(output_folder,train_bin_file_name), "create")
    validateStore = kvstore.FileStore()
    validateStore.open(os.path.join(output_folder,validate_bin_file_name), "create")
    testStore = kvstore.FileStore()
    testStore.open(os.path.join(output_folder,test_bin_file_name), "create")
    
    meanStore = kvstore.FileStore()
    meanStore.open(os.path.join(output_folder,mean_bin_file_name), "create")
    
 
    count=0
    trainCount=0
    validateCount=0
    testCount=0

    # the expected image binary length
    binaryLength=3*size[0]*size[1] 

    meanRecord = RecordProto()
    meanRecord.shape.extend([3,size[0],size[1]])
    for i in range(binaryLength):
        meanRecord.data.append(0.0)

    for (label,f) in fileList:    

        im =Image.open(f)
        #the image size should be equal
        assert im.size==size

        binaryContent=str(toBin(im,size))

        count +=1
        record = RecordProto()
        record.shape.extend([3,size[0],size[1]])
        record.label=label
        record.pixel=binaryContent

        value = record.SerializeToString()
    
        if count <= train_num :
            key = "%05d" % trainCount 
            trainCount+=1
            trainStore.write(key,value) 
            #only caculate train data's mean
            for i in range(binaryLength):
                meanRecord.data[i]+=ord(binaryContent[i])
        elif count <= train_num+validate_num :
            key = "%05d" % validateCount 
            validateCount+=1
            validateStore.write(key,value) 
        elif count <= train_num+validate_num+test_num:
            key = "%05d" % testCount 
            testCount+=1
            testStore.write(key,value) 
        else:
            break

    for i in range(binaryLength):
        meanRecord.data[i]/=trainCount

    meanStore.write("mean", meanRecord.SerializeToString())
    meanStore.flush()
    meanStore.close()
     
    trainStore.flush()
    trainStore.close()
    validateStore.flush()
    validateStore.close()
    testStore.flush()
    testStore.close()

    meta_file.write("image size: "+str(size[0])+"*"+str(size[1])+"\n")    
    meta_file.write("total file num: "+str(count)+"\n")    
    meta_file.write("train file num: "+str(trainCount)+"\n")
    meta_file.write("validate file num: "+str(validateCount)+"\n")
    meta_file.write("test file num: "+str(testCount)+"\n")
    meta_file.write("label list:[\n")

    for item in labelList:
        meta_file.write("("+str(item[0])+",\""+item[1]+"\"),\n")
    meta_file.write("]")
    meta_file.flush()
    meta_file.close()

    print "end at: "+time.strftime('%X %x %Z')    

    return labelList
