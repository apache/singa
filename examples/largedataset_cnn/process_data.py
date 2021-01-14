import os
import imghdr
import pickle as pkl
import numpy as np
from PIL import Image

def paths_to_images(paths, image_size):
    num_images=len(paths)
    im = np.zeros((num_images,3,image_size,image_size), dtype=np.float32)

    for i in range(num_images):
        temp = np.array(Image.open(paths[i]).convert('RGB').resize((image_size, image_size), Image.BILINEAR))
        temp = np.moveaxis(temp,-1,0)
        im[i] = temp  

    im /= 255

    return im


def process_data(dataset_root, classes):
  # load class names
    with open(classes, 'r', encoding='utf-8') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
        num_classes = len(classes)

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in os.listdir(dataset_root):
        class_root = os.path.join(dataset_root, class_name)
        class_id = classes.index(class_name)
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) is None:
                # this is not an image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    # convert to numpy array
    input_paths = np.array(input_paths)
    labels = np.array(labels, dtype=np.int32)

    # shuffle dataset
    np.random.seed(0)
    perm = np.random.permutation(len(input_paths))
    input_paths = input_paths[perm]
    labels = labels[perm]

    # split dataset for training and validation
    border = int(len(input_paths) * 0.8)
    train_labels = labels[:border]
    val_labels = labels[border:]
    train_input_paths = input_paths[:border]
    val_input_paths = input_paths[border:]
    

    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))
    
    return train_input_paths, train_labels, val_input_paths, val_labels

def loaddata():
    dataset_root = '/Dataset/Data/'
    classes = '/Dataset/classes.txt'    
    return process_data(dataset_root, classes)

if __name__ == '__main__':

    # test script in main
    train_input_paths, train_labels, val_input_paths, val_labels = loaddata()

    print(train_input_paths.shape)
    print(train_labels.shape)
    print(val_input_paths.shape)
    print(val_labels.shape)   

    a=paths_to_images(paths=train_input_paths[0:5], image_size=299)
    print(a)
    print(a.shape)
    