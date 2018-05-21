# 数据(Data)

这个模块包含加载和预获取批数据的类。

示例用法：

```python
import image_tool
from PIL import Image

tool = image_tool.ImageTool()

def image_transform(img_path):
    global tool
    return tool.load(img_path).resize_by_range(
        (112, 128)).random_crop(
        (96, 96)).flip().get()

data = ImageBatchIter('train.txt', 3,
                      image_transform, shuffle=True, delimiter=',',
                      image_folder='images/',
                      capacity=10)
data.start()
# imgs is a numpy array for a batch of images,
# shape: batch_size, 3 (RGB), height, width
imgs, labels = data.next()

# convert numpy array back into images
for idx in range(imgs.shape[0]):
    img = Image.fromarray(imgs[idx].astype(np.uint8).transpose(1, 2, 0),
                          'RGB')
    img.save('img%d.png' % idx)
data.end()
```

---

### class singa.data.ImageBatchIter(img_list_file, batch_size, image_transform, shuffle=True, delimiter=' ', image_folder=None, capacity=10)

迭代地从数据集中获取批数据。

**参数：**
- **img_list_file (str)** – 包含源数据的文件名；每行包含image_path_suffix和标签
- **batch_size (int)** – 每个mini-bach包含的样本数目
- **image_transform** – 图像增强函数；它接受完整的图像路径并输出一系列增强后的图像
- **shuffle (boolean)** – 为真表示对列表做搅乱
- **delimiter (char)** – image_path_suffix和标签之间的分割符, 例如空格或逗号
- **image_folder (boolean)** – 图片路径的前缀
- **capacity (int)** – 内部队列的最大mini-batch数目

---
