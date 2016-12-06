# Use parameters pre-trained from Caffe in SINGA

In this example, we use SINGA to load the VGG parameters trained by Caffe to do image classification.

## Run this example
You can run this example by simply executing `run.sh vgg16` or `run.sh vgg19`
The script does the following work.

### Obtain the Caffe model
* Download caffe model prototxt and parameter binary file.
* Currently we only support the latest caffe format, if your model is in
    previous version of caffe, please update it to current format.(This is
    supported by caffe)
* After updating, we can obtain two files, i.e., the prototxt and parameter
    binary file.

### Prepare test images
A few sample images are downloaded into the `test` folder.

### Predict
The `predict.py` script creates the VGG model and read the parameters,

    usage: predict.py [-h] model_txt model_bin imgclass

where `imgclass` refers to the synsets of imagenet dataset for vgg models.
You can start the prediction program by executing the following command:

    python predict.py ./vgg16.prototxt ./vgg16.caffemodel ./synset_words.txt

Then you type in the image path, and the program would output the top-5 labels.

**NOTE** Caffe uses OpenCV to load images, whose channels are in BGR order. To
do prediction using Caffe's pre-trained parameters, please make sure the input
tensor of SINGA net is in BGR order. cv2 loads images as BGR, hence you don't need
to do any additional processing. If you use pillow to read images, you need to
swap the channel as shown in predict.py.


More Caffe models would be tested soon.
