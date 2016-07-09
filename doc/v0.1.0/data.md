# Data Preparation

---

To submit a training job, users need to convert raw data (e.g., images, text
documents) into SINGA recognizable [Record](../api-v0.1.0/classsinga_1_1Record.html)s.
SINGA uses [data layers](layer#data-layers)
to load these records into memory and uses
[parser layers](layer#parser-layers) to parse features (e.g.,
image pixels and labels) from these `Record`s. `Record`s could be
stored in a file, a database, or HDFS, as
long as there is a corresponding
[DataLayer](../api-v0.1.0/classsinga_1_1DataLayer.html).

## DataShard

SINGA comes with a light-weight database named [DataShard](../api-v0.1.0/classsinga_1_1DataShard.html).
It provides operations for inserting `Record`,
and read `Record` in sequential order.
`Record`s are flushed once the maximum cache size is reached. It
loads `Record`s in batch and returns them to users one by one through the
[Next](../api-v0.1.0/classsinga_1_1DataShard.html) function.
The disk folder in which the `Record`s are stored, is called a (data) shard. The
[ShardDataLayer](../api-v0.1.0/classsinga_1_1ShardDataLayer.html) is a built-in
layer for loading `Record`s from `DataShard`.

To create data shards for users' own data, they can follow the subsequent sections.

###  User record definition

Users define their own record for storing their data. E.g., the built-in
[SingleLabelImageRecord](../api-v0.1.0/classsinga_1_1SingleLabelImageRecord.html)
has an int field for image label, and a pixel array for image RGB values.
The code below shows an example of defining a new record `UserRecord`, and extending the
base `Record` to include `UserRecord`.


    package singa;

    import "common.proto";  // required to import common.proto

    message UserRecord {
        repeated int userVAR1 = 1;    // unique field id
        optional string userVAR2 = 2; // unique field id
        ...
    }

    extend Record {
        optional UserRecord user_record = 101;  // unique extension field id, reserved for users (e.g., 101-200)
    }

Please refer to the
[Tutorial](https://developers.google.com/protocol-buffers/docs/reference/cpp-generated?hl=en#extension)
for extension of protocol messages.

The extended `Record` will be parsed by a parser layer to extract features
(e.g., label or pixel values). Users need to write
their own [parser layers](layer#parser-layers) to parse the
extended `Record`.


*Note*

There is an alternative way to define the proto extension.
In this way, you should be careful of the scope of fields and how to access the
fields, which are different from the above.

    message UserRecord {
        extend Record {
            optional UserRecord user_record = 101;  // unique extension field id, reserved for users (e.g., 101-200)
        }
        repeated int userVAR1 = 1; // unique field id
        optional string userVAR2 = 2; // unique field id
        ...
    }

###  DataShard creation

Users write code to convert their data into `Record`s and insert them into shards
following the subsequent steps.

1. Create a folder *USER_DATA* under *SINGA_ROOT*.

2. Prepare the source file, e.g., `create_shard.cc`,  in `SINGA_ROOT/USER_DATA`

        singa::DataShard myShard(outputpath, kCreate);
        // outputpath is the path of the folder for storing the shard

    the above code opens a folder for storing the data shard.

        singa::Record record;
        singa::UserRecord* r = record.MutableExtension(singa::user_record);

    an user-defined record is allocated by the above code.

        r->add_userVAR1( int_val );     // for repeated field
        r->set_userVAR2( string_val );

    users load raw data and set/add them into user-defined record as shown above.

        // key (string) is a unique record ID (e.g., converted from a number starting from 0)
        myShard.Insert( key, record );

    Once the `record` object is filled, it is inserted into the shard as shown above.
    If there are multiple data records, they should be inserted sequentially.
    After inserting all records, the shard is created into the `outputpath` folder.

3. Compile and link. Both *user.proto* and *create.cc* should be compiled and linked with libsinga.so.
  The following instruction generates *user.pb.cc* and *user.pb.h* from *user.proto*.

        protoc -I=SINGA_ROOT/USER_DATA --cpp_out=SINGA_ROOT/USER_DATA user.proto

    All code can be compiled and linked into an executable file

        g++ create_shard.cc user.pb.cc -std=c++11 -lsinga \
          -ISINGA_ROOT/include -LSINGA_ROOT/.libs/ -Wl,-unresolved-symbols=ignore-in-shared-libs \

          -Wl,-rpath=SINGA_ROOT/.libs/  -o create_shard.bin


4. Run the program. Once the executable file is generated, users can run it to create data shards.

        ./create_shard.bin  <args>



### Example - CIFAR dataset

This example uses the [CIFAR-10 image dataset](http://www.cs.toronto.edu/~kriz/cifar.html) collected by Alex Krizhevsky.
It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
There are 50,000 training images and 10,000 test images.
Each image has a single label. This dataset is stored in binary files with specific format.
SINGA has written the [create_shard.cc](https://github.com/apache/incubator-singa/blob/master/examples/cifar10/create_shard.cc)
to convert images in the binary files into `Record`s and insert them into training and test shards.

1. Download raw data. The following command will download the dataset into *cifar-10-batches-bin* folder.

        # in SINGA_ROOT/examples/cifar10
        $ cp Makefile.example Makefile   // an example makefile is provided
        $ make download

2. Since `Record` already has one `image` field which is designed for
  single-label images, e.g., images from CIFAR10, we can use it directly.
  Particularly, the type of `image` is `SingleLabelImageRecord`,


        # common.proto
        package singa;

        message Record {
          enum Type {
            kSingleLabelImage = 0;
          }
          optional Type type = 1 [default = kSingleLabelImage];
          optional SingleLabelImageRecord image = 2;   // for configuration
        }

        message SingleLabelImageRecord {
          repeated int32 shape = 1;                // it obtains 3 (rgb channels), 32 (row), 32 (col)
          optional int32 label = 2;                // label
          optional bytes pixel = 3;                // pixels
          repeated float data = 4 [packed = true]; // it is used for normalization
        }

3. Add/Set data into the record, and insert it to shard.
    `create_shard.cc` reads images (and labels) from the downloaded binary files.
    For each image, it puts the image feature and label into a `SingleLabelImageRecord`
    of `Record`, and then inserts the `Record` into `DataShard`.

          ...// open binary files
        DataShard train_shard("cifar10_train_datashard", DataShard::kCreate);

        singa::Record record;
        singa::SingleLabelImageRecord* image = record.mutable_image();;
        for (int image_id = 0; image_id < 50000; image_id ++) {
              read_image(&data_file, &label, str_buffer);  // read feature and label from binary file
              image->set_label(label);  // put label
              image->set_pixel(str_buffer);  // put image feature
              train_shard.Insert(to_string(image_id), record);  // insert a record with unique ID
        }

    The data shard for testing data is created similarly.
    In addition, it computes average values (not shown here) of image pixels as another `Record`
    which is directly serialized into *SINGA_ROOT/USER_DATA/image_mean.bin*.
    The mean values will be used for preprocessing image features.

        for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
          const string& pixels = image->pixel();
          for(int i=0; i<kCIFARImageNBytes; i++)
            mean.set_data(i, mean.data(i)+static_cast<uint8_t>(pixels[i]));
          count += 1;
        }
        for(int i=0; i<kCIFARImageNBytes; i++)
          mean.set_data(i, mean.data(i)/count);

4. Compile and run the program. SINGA provides an example Makefile that contains instructions
    for compiling the source code and linking it with *libsinga.so*. Users just execute the following command.

        $ make create

    The data shards for training and testing will be generated into
    *cifar10_train_shard* and *cifar10_test_shard* folders respectively.

### Example - MNIST dataset

This example creates `DataShard`s for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
It has a training set of 60,000 handwritten digit images, and a test set of 10,000 images.
Similar to the images of CIFAR10, each MNIST image has a single label. Hence, we still
use the built-in `Record`. The process is almost the same as that for
the CIFAR10 dataset, except that the MNIST dataset is downloaded as binary files with
another format. SINGA has written the *create_shard.cc* program to parse the binary files
and convert MNIST images into `Record`s.

The following command will download the dataset

    $ cp Makefile.example Makefile   // an example makefile is provided
    $ make download

Data shards will be generated into *mnist_train_shard* and *mnist_test_shard* by

    $ make create


## LMDB

To be filled soon.

## HDFS

To be filled soon.

