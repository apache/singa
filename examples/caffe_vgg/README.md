#Convert model parameter of caffe to singa

## Run this example
You can run this example by simply execute `run.sh vgg16` or `run.sh vgg19`

## Obtain caffe model
* Download caffe model prototxt and parameter binary file. 
* Currently we only support the latest caffe format, if your model is in
    previous version of caffe, please update it to current format.(This is
    supported by caffe)
* After updating, we can obtain two files, i.e., the prototxt and parameter
    binary file.

## Prepare test images
As an example, we just put several images in a single folder, for example,
`./test`, the program will load all images in the folder and will run to test.

## Predict
Run `predict.py`
`usage: predict.py [-h] model_txt model_bin imgclass testdir`
where `imgclass` refers to the synsets of imagenet dataset for vgg models.
In vgg example, you can run it by executing the following command:
`python predict.py ./vgg16.prototxt ./vgg16.caffemodel ./synset_words.txt ./test`
