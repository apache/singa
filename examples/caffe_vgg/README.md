#Convert model parameter of caffe to singa
## Obtain caffe model
* Download caffe model prototxt and parameter binary file. Here you can
    download vgg model by run `download.sh`, e.g.,
    `sh download.sh vgg16` or `sh download.sh vgg19`
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
