name: DenseNet on ImageNet SINGA version: 1.1.1 SINGA commit: license: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

---

# 用DenseNet做图像分类


这个例子中，我们将PyTorch训练好的DenseNet转换为SINGA模型以用作图像分类。

## 操作说明

* 下载参数的checkpoint文件到如下目录

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-121.tar.gz
		$ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/synset_words.txt
		$ tar xvf densenet-121.tar.gz

* 运行程序

        $ python serve.py -h
		
* 运行程序

        # use cpu
		$ python serve.py --use_cpu --parameter_file densenet-121.pickle --depth 121 &
		# use gpu
		$ python serve.py --parameter_file densenet-121.pickle --depth 121 &

* 提交图片进行分类

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg和image3.jpg应该在执行指令前就已被下载。

## 详细信息

用`convert.py`从Pytorch参数文件中提取参数值

* 运行程序

    	$ python convert.py -h