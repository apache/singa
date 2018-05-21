# 图像工具

图像增强的模型。

示例用法：

```python
from singa import image_tool

tool = image_tool.ImageTool()
imgs = tool.load('input.png').resize_by_list([112]).crop5((96, 96), 5).enhance().flip().get()
for idx, img in enumerate(imgs):
    img.save('%d.png' % idx)
```

---

### class singa.image_tool.ImageTool

一个图像增强工具。 对于inplace = True的操作，返回的值是ImageTool实例，用于链接多个操作; 否则，将返回预处理过的图像。 对于具有可数预处理情况的操作，可以设置参数num_case来决定要应用的预处理情况的数量。 通常，训练阶段设置为1，测试阶段设置为最大。

---

#### color_cast(offset=20, inplace=True)

对每个通道加上一个随机偏移值[-offset, offset]。

**参数：**
- **offset** – 偏移， >0 and <255
- **inplace** – 对原图对象操作或返回一张新图

---

#### crop3(patch, num_case=1, inplace=True)

对给定位置，截取可能的最大方框并缩放到给定尺寸。 按照图像尺寸，截取位置可以是（左， 中， 右）或（上， 中， 下）之一。

**参数：**
- **patch (tuple)** – 输出图像的高和宽
- **num_case** – 情况数目, 必须在[1,3]
- **inplace** – 对原图对象操作或返回一张新图

---

#### crop5(patch, num_case=1, inplace=True)

截取位置可以是[左上， 左下， 右上， 右下， 中间]。

**参数：**
- **patch (tuple)** – 输出图像的高和宽
- **num_case** – 情况数目, 必须在[1,5]
- **inplace** – 对原图对象操作或返回一张新图

---

#### crop8(patch, num_case=1, inplace=True)

这是patch_5和patch_and_scale的并集。 你可以依照这个例子取任何情况的并集。

---

#### enhance(scale=0.2, inplace=True)

对色度、对比度、亮度和锐度采用随机增强。

**参数：**
- **scale (float)** – 增强范围 [1-scale, 1+scale]
- **inplace** – 对原图对象操作或返回一张新图

---

#### flip(num_case=1, inplace=True)

随机向左或向右翻转图像。

**参数：**
- **num_case** – 情况数目，必须是 {1,2}; 如果是2，则会返回原图以及翻转的图像。
- **inplace** – 对原图对象操作或返回一张新图

---

#### num_augmentation()

返回每张图像被增强后的总数

---

#### random_crop(patch, inplace=True)

根据随机偏移截取指定大小的图像

**参数：**
- **patch (tuple)** – 截取图像块的高和宽
- **inplace (Boolean)** – 如果为真，直接用新图像块替换原始图像内容；否则返回新图像块。

---

#### resize_by_list(size_list, num_case=1, inplace=True)

**参数：**
- **num_case** – 缩放操作数目， 必须不超过size_list的长度
- **inplace** – 对原图对象操作或返回一张新图

---

#### resize_by_range(rng, inplace=True)

**参数：**
- **rng** – 元组 (起始值, 结束值), 包括起始值但不包括结束值
- **inplace** – 对原图对象操作或返回一张新图

---

#### rotate_by_list(angle_list, num_case=1, inplace=True)

**参数：**
- **num_case** – 旋转操作数目， 必须不超过angle_list的长度
- **inplace** – 对原图对象操作或返回一张新图

---

#### rotate_by_range(rng, inplace=True)

**参数：**
- **rng** – 表示旋转角度范围的元组 (起始值, 结束值), 包括起始值但不包括结束值
- **inplace** – 对原图对象操作或返回一张新图

---

#### singa.image_tool.color_cast(img, offset)

对每个通道加上一个随机偏移值[-offset, offset]。

---

#### singa.image_tool.crop(img, patch, position)

截取给定位置和给定大小的图像块。

**参数：**
- **patch (tuple)** – 宽，高
- **position (list(str))** – 左上，左下，右上，右下，中间

---

#### singa.image_tool.crop_and_resize(img, patch, position)

对给定位置，截取可能的最大方框并缩放到给定尺寸。

**参数：**
- **patch (tuple)** – 宽，高
- **position (list(str))** – 左，中，右，上，中，下

---

#### singa.image_tool.enhance(img, scale)

对色度、对比度、亮度和锐度采用随机增强。

**参数：**
- **scale (float)** – 增强范围 [1-scale, 1+scale]

---

#### singa.image_tool.load_img(path, grayscale=False)

从给定路径读取图像。

---

#### singa.image_tool.resize(img, small_size)

缩放图像使得最短边达到给定大小。

---
