#include <gtest/gtest.h>
#include <sys/stat.h>
#include <cstdint>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "model/layer.h"
#include "proto/model.pb.h"
#include "utils/shard.h"
using namespace singa;
TEST(MnistLayerTest, SingleScale){
  LayerProto proto;
  MnistProto *mnist=proto.mutable_mnist_param();
  mnist->set_size(55);
  MnistImageLayer layer;
  layer.FromProto(proto);
  cv::Mat image;
  image=cv::imread("src/test/data/mnist.png", 0);
  string pixel;
  pixel.resize(image.rows*image.cols);
  for(int i=0,k=0;i<image.rows;i++)
    for(int j=0; j<image.cols;j++)
      pixel[k++]=static_cast<char>(image.at<uint8_t>(i,j));
  Record rec;
  rec.set_type(Record_Type_kMnist);
  MnistRecord *mrec=rec.mutable_mnist();
  mrec->set_pixel(pixel);
  layer.Setup(1, rec, kNone);
  layer.AddInputRecord(rec);

  const vector<uint8_t>& dat=layer.Convert2Image(0);
  int s=static_cast<int>(sqrt(dat.size()));
  cv::Mat newimg(s,s,CV_8UC1);
  int count=0;
  for(int i=0,k=0;i<newimg.rows;i++)
    for(int j=0; j<newimg.cols;j++){
      count+=dat[k]>0;
      newimg.at<uint8_t>(i,j)=dat[k++];
    }
  //LOG(ERROR)<<"image positive "<<count<<" size "<<s;
  cv::imwrite("src/test/data/mnist_scale.png", newimg);
}

TEST(MnistLayerTest, SingleAffineTransform){
  LayerProto proto;
  MnistProto *mnist=proto.mutable_mnist_param();
  mnist->set_beta(15);
  mnist->set_gamma(16);
  mnist->set_size(55);
  MnistImageLayer layer;
  layer.FromProto(proto);
  cv::Mat image;
  image=cv::imread("src/test/data/mnist.png", 0);
  string pixel;
  pixel.resize(image.rows*image.cols);
  for(int i=0,k=0;i<image.rows;i++)
    for(int j=0; j<image.cols;j++)
      pixel[k++]=static_cast<char>(image.at<uint8_t>(i,j));
  Record rec;
  rec.set_type(Record_Type_kMnist);
  MnistRecord *mrec=rec.mutable_mnist();
  mrec->set_pixel(pixel);
  layer.Setup(1, rec, kNone);
  layer.AddInputRecord(rec);

  const vector<uint8_t>& dat=layer.Convert2Image(0);
  int s=static_cast<int>(sqrt(dat.size()));
  cv::Mat newimg(s,s,CV_8UC1);
  int count=0;
  for(int i=0,k=0;i<newimg.rows;i++)
    for(int j=0; j<newimg.cols;j++){
      count+=dat[k]>0;
      newimg.at<uint8_t>(i,j)=dat[k++];
    }
  //LOG(ERROR)<<"image positive "<<count<<" size "<<s;

  cv::imwrite("src/test/data/mnist_affine.png", newimg);
}
TEST(MnistLayerTest, SingleElasticDistortion){
  LayerProto proto;
  MnistProto *mnist=proto.mutable_mnist_param();
  mnist->set_elastic_freq(1);
  mnist->set_sigma(6);
  mnist->set_alpha(36);
  mnist->set_beta(15);
  mnist->set_gamma(16);
  mnist->set_size(55);
  mnist->set_kernel(21);
  MnistImageLayer layer;
  layer.FromProto(proto);
  cv::Mat image;
  image=cv::imread("src/test/data/mnist.png", 0);
  string pixel;
  pixel.resize(image.rows*image.cols);
  for(int i=0,k=0;i<image.rows;i++)
    for(int j=0; j<image.cols;j++)
      pixel[k++]=static_cast<char>(image.at<uint8_t>(i,j));
  Record rec;
  rec.set_type(Record_Type_kMnist);
  MnistRecord *mrec=rec.mutable_mnist();
  mrec->set_pixel(pixel);
  layer.Setup(1, rec, kNone);
  layer.AddInputRecord(rec);

  const vector<uint8_t>& dat=layer.Convert2Image(0);
  int s=static_cast<int>(sqrt(dat.size()));
  cv::Mat newimg(s,s,CV_8UC1);
  int count=0;
  for(int i=0,k=0;i<newimg.rows;i++)
    for(int j=0; j<newimg.cols;j++){
      count+=dat[k]>0;
      newimg.at<uint8_t>(i,j)=dat[k++];
    }
  cv::imwrite("src/test/data/mnist_elastic.png", newimg);
}
TEST(MnistLayerTest, MultElasticDistortion){
  LayerProto proto;
  MnistProto *mnist=proto.mutable_mnist_param();
  int kTotal=100;
  int kSize=29;
  mnist->set_elastic_freq(kTotal);
  mnist->set_sigma(6);
  mnist->set_alpha(36);
  mnist->set_beta(15);
  mnist->set_gamma(16);
  mnist->set_size(kSize);
  mnist->set_kernel(21);
  MnistImageLayer layer;
  layer.FromProto(proto);
  vector<vector<int>> shapes{{kTotal, kSize,kSize}};
  layer.Setup(shapes, kNone);
  shard::Shard source("/data1/wangwei/singa/data/mnist/test/",shard::Shard::kRead);
  int n=static_cast<int>(sqrt(kTotal));
  cv::Mat origin(n*28,n*28, CV_8UC1);
  char disp[1024];
  for(int x=0;x<n;x++){
    sprintf(disp+strlen(disp), "\n");
    for(int y=0;y<n;y++){
      Record rec;
      string key;
      CHECK(source.Next(&key, &rec));
      const string pixel=rec.mnist().pixel();
      cv::Mat img=origin(cv::Rect(y*28, x*28, 28, 28));
      for(int i=0,k=0;i<28;i++)
        for(int j=0;j<28;j++)
          img.at<uint8_t>(i,j)=static_cast<uint8_t>(pixel[k++]);
      layer.AddInputRecord(rec);
      sprintf(disp+strlen(disp), "%d ", rec.mnist().label());
    }
  }
  LOG(ERROR)<<disp;
  cv::imwrite("src/test/data/mnist_big.png", origin);

  cv::Mat output(n*kSize,n*kSize, CV_8UC1);
  for(int i=0;i<kTotal;i++){
    const vector<uint8_t>& dat=layer.Convert2Image(i);
    int x=(i/n);
    int y=i%n;
    cv::Mat img=output(cv::Rect(y*kSize, x*kSize, kSize, kSize));
    for(int i=0,k=0;i<kSize;i++)
      for(int j=0;j<kSize;j++)
        img.at<uint8_t>(i,j)=dat[k++];
  }
  cv::imwrite("src/test/data/mnist_bigout.png", output);
}
