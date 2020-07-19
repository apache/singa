# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
'''
An utility model for image augmentation.

Example usage::

    from singa import image_tool

    tool = image_tool.ImageTool()
    imgs = tool.load('input.png').\
        resize_by_list([112]).crop5((96, 96), 5).enhance().flip().get()
    for idx, img in enumerate(imgs):
        img.save('%d.png' % idx)

'''
from __future__ import division

from builtins import range
from builtins import object
import random
import numpy as np
from PIL import Image, ImageEnhance
import math


def load_img(path, grayscale=False):
    '''Read the image from a give path'''
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    return img


def crop(img, patch, position):
    '''Crop the input image into given size at given position.

    Args:
        patch(tuple): width and height of the patch
        position(list(str)): left_top, left_bottom, right_top, right_bottom
        and center.
    '''
    if img.size[0] < patch[0]:
        raise Exception('img size[0] %d is smaller than patch[0]: %d' %
                        (img.size[0], patch[0]))
    if img.size[1] < patch[1]:
        raise Exception('img size[1] %d is smaller than patch[1]: %d' %
                        (img.size[1], patch[1]))

    if position == 'left_top':
        left, upper = 0, 0
    elif position == 'left_bottom':
        left, upper = 0, img.size[1] - patch[1]
    elif position == 'right_top':
        left, upper = img.size[0] - patch[0], 0
    elif position == 'right_bottom':
        left, upper = img.size[0] - patch[0], img.size[1] - patch[1]
    elif position == 'center':
        left, upper = (img.size[0] - patch[0]) // 2, (img.size[1] -
                                                      patch[1]) // 2
    else:
        raise Exception('position is wrong')

    box = (left, upper, left + patch[0], upper + patch[1])
    new_img = img.crop(box)
    # print "crop to box %d,%d,%d,%d" % box
    return new_img


def crop_and_resize(img, patch, position):
    '''Crop a max square patch of the input image at given position and resize
    it into given size.

    Args:
        patch(tuple): width, height
        position(list(str)): left, center, right, top, middle, bottom.
    '''
    size = img.size
    if position == 'left':
        left, upper = 0, 0
        right, bottom = size[1], size[1]
    elif position == 'center':
        left, upper = (size[0] - size[1]) // 2, 0
        right, bottom = (size[0] + size[1]) // 2, size[1]
    elif position == 'right':
        left, upper = size[0] - size[1], 0
        right, bottom = size[0], size[1]
    elif position == 'top':
        left, upper = 0, 0
        right, bottom = size[0], size[0]
    elif position == 'middle':
        left, upper = 0, (size[1] - size[0]) // 2
        right, bottom = size[0], (size[1] + size[0]) // 2
    elif position == 'bottom':
        left, upper = 0, size[1] - size[0]
        right, bottom = size[0], size[1]
    else:
        raise Exception('position is wrong')
    box = (left, upper, right, bottom)
    new_img = img.crop(box)

    new_img = new_img.resize(patch, Image.BILINEAR)
    # print box+crop
    # print "crop to box %d,%d,%d,%d and scale to %d,%d" % (box+crop)
    return new_img


def resize(img, small_size):
    '''Resize the image to make the smaller side be at the given size'''
    size = img.size
    if size[0] < size[1]:
        new_size = (small_size, int(small_size * size[1] / size[0]))
    else:
        new_size = (int(small_size * size[0] / size[1]), small_size)
    new_img = img.resize(new_size, Image.BILINEAR)
    # print 'resize to (%d,%d)' % new_size
    return new_img


def color_cast(img, offset):
    '''Add a random value from [-offset, offset] to each channel'''
    x = np.asarray(img, dtype='uint8')
    x.flags.writeable = True
    cast_value = [0, 0, 0]
    for i in range(3):
        r = random.randint(0, 1)
        if r:
            cast_value[i] = random.randint(-offset, offset)
    for w in range(x.shape[0]):
        for h in range(x.shape[1]):
            for c in range(3):
                if cast_value[c] == 0:
                    continue
                v = x[w][h][c] + cast_value[c]
                if v < 0:
                    v = 0
                if v > 255:
                    v = 255
                x[w][h][c] = v
    new_img = Image.fromarray(x.astype('uint8'), 'RGB')
    return new_img


def enhance(img, scale):
    '''Apply random enhancement for Color,Contrast,Brightness,Sharpness.

    Args:
        scale(float): enhancement degree is from [1-scale, 1+scale]
    '''
    enhance_value = [1.0, 1.0, 1.0, 1.0]
    for i in range(4):
        r = random.randint(0, 1)
        if r:
            enhance_value[i] = random.uniform(1 - scale, 1 + scale)
    if not enhance_value[0] == 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(enhance_value[0])
    if not enhance_value[1] == 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(enhance_value[1])
    if not enhance_value[2] == 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(enhance_value[2])
    if not enhance_value[3] == 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(enhance_value[3])
    return img


def flip(img):
    # print 'flip'
    new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return new_img


def flip_down(img):
    # print 'flip_down'
    new_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return new_img


def get_list_sample(l, sample_size):
    return [l[i] for i in sorted(random.sample(range(len(l)), sample_size))]


class ImageTool(object):
    '''A tool for image augmentation.

    For operations with inplace=True, the returned value is the ImageTool
    instance self, which is for chaining multiple operations; Otherwise, the
    preprocessed images would be returned.

    For operations that has countable pre-processing cases, argument num_case
    could be set to decide the number of pre-processing cases to apply.
    Typically, it is set to 1 for training phases and to the max for test
    phases.
    '''

    def __init__(self):
        self.imgs = []
        return

    def load(self, path, grayscale=False):
        img = load_img(path, grayscale)
        self.imgs = [img]
        return self

    def set(self, imgs):
        self.imgs = imgs
        return self

    def append(self, img):
        self.imgs.append(img)
        return self

    def get(self):
        return self.imgs

    def num_augmentation(self):
        '''Return the total number of augmentations to each image'''
        pass

    def resize_by_range(self, rng, inplace=True):
        '''
        Args:
            rng: a tuple (begin,end), include begin, exclude end
            inplace: inplace imgs or not ( return new_imgs)
        '''
        size_list = list(range(rng[0], rng[1]))
        return self.resize_by_list(size_list, 1, inplace)

    def resize_by_list(self, size_list, num_case=1, inplace=True):
        '''
        Args:
            num_case: num of resize cases, must be <= the length of size_list
            inplace: inplace imgs or not ( return new_imgs)
        '''
        new_imgs = []
        if num_case < 1 or num_case > len(size_list):
            raise Exception(
                'num_case must be smaller in [0,%d(length of size_list)]' %
                len(size_list))
        for img in self.imgs:
            if num_case == len(size_list):
                small_sizes = size_list
            else:
                small_sizes = get_list_sample(size_list, num_case)

            for small_size in small_sizes:
                new_img = resize(img, small_size)
                new_imgs.append(new_img)
        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def rotate_by_range(self, rng, inplace=True):
        '''
        Args:
            rng: a tuple (begin,end) in degree, include begin, exclude end
            inplace: inplace imgs or not ( return new_imgs)
        '''
        angle_list = list(range(rng[0], rng[1]))
        return self.rotate_by_list(angle_list, 1, inplace)

    def rotate_by_list(self, angle_list, num_case=1, inplace=True):
        '''
        Args:
            num_case: num of rotate cases, must be <= the length of angle_list
            inplace: inplace imgs or not ( return new_imgs)
        '''
        new_imgs = []
        if num_case < 1 or num_case > len(angle_list):
            raise Exception(
                'num_case must be smaller in [1,%d(length of angle_list)]' %
                len(angle_list))

        for img in self.imgs:
            if num_case == len(angle_list):
                angles = angle_list
            else:
                angles = get_list_sample(angle_list, num_case)

            for angle in angles:
                new_img = img.rotate(angle)
                new_imgs.append(new_img)
        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def crop5(self, patch, num_case=1, inplace=True):
        '''Crop at positions from [left_top, left_bottom, right_top,
        right_bottom, and center].

        Args:
            patch(tuple): width and height of the result image.
            num_case: num of cases, must be in [1,5]
            inplace: inplace imgs or not ( return new_imgs)
        '''
        new_imgs = []
        positions = [
            "left_top", "left_bottom", "right_top", "right_bottom", "center"
        ]
        if num_case > 5 or num_case < 1:
            raise Exception('num_case must be in [1,5]')
        for img in self.imgs:

            if num_case < 5:
                positions = get_list_sample(positions, num_case)

            for position in positions:
                new_img = crop(img, patch, position)
                new_imgs.append(new_img)

        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def crop3(self, patch, num_case=1, inplace=True):
        '''Crop a max square patch of the input image at given position and
        scale it into given size.

        According to img size, crop position could be either
        (left, center, right) or (top, middle, bottom).

        Args:
            patch(tuple): the width and height the output image
            num_case: num of cases, must be in [1,3]
            inplace: inplace imgs or not ( return new_imgs)
        '''
        if not patch[0] == patch[1]:
            raise Exception('patch must be a square')
        new_imgs = []
        if num_case > 3 or num_case < 1:
            raise Exception('num_case must be in [1,3]')
        positions_horizental = ["left", "center", "right"]
        positions_vertical = ["top", "middle", "bottom"]
        for img in self.imgs:
            size = img.size
            if size[0] > size[1]:
                if num_case < 3:
                    positions = get_list_sample(positions_horizental, num_case)
                else:
                    positions = positions_horizental
            else:
                if num_case < 3:
                    positions = get_list_sample(positions_vertical, num_case)
                else:
                    positions = positions_vertical

            for position in positions:
                new_img = crop_and_resize(img, patch, position)
                new_imgs.append(new_img)

        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def crop8(self, patch, num_case=1, inplace=True):
        '''This is a union of patch_5 and patch_and_scale.

        You can follow this example to union any num of cases of imgtool methods
        '''
        patch5 = 5
        patch3 = 3
        if num_case < 1 or num_case > patch5 + patch3:
            raise Exception('num_case must be in [0,%d]' % (patch5 + patch3))
        if num_case == patch5 + patch3:
            count = patch5
        else:
            sample_list = list(range(0, patch5 + patch3))
            samples = get_list_sample(sample_list, num_case)
            count = 0
            for s in samples:
                if s < patch5:
                    count += 1
        new_imgs = []
        if count > 0:
            new_imgs += self.crop5(patch, count, False)
        if num_case - count > 0:
            new_imgs += self.crop3(patch, num_case - count, False)

        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def random_crop(self, patch, inplace=True):
        '''Crop the image at random offset to get a patch of the given size.

        Args:
            patch(tuple): width and height of the patch
            inplace(Boolean): replace the internal images list with the patches
                              if True; otherwise, return the patches.
        '''
        new_imgs = []
        for img in self.imgs:
            assert img.size[0] >= patch[0] and img.size[1] >= patch[1],\
                'img size (%d, %d), patch size (%d, %d)' % \
                (img.size[0], img.size[1], patch[0], patch[1])
            left_offset = random.randint(0, img.size[0] - patch[0])
            top_offset = random.randint(0, img.size[1] - patch[1])
            box = (left_offset, top_offset, left_offset + patch[0],
                   top_offset + patch[1])
            new_imgs.append(img.crop(box))

        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def random_crop_resize(self, patch, inplace=True):
        ''' Crop of the image at a random size between 0.08 to 1 of input image
            and random aspect ratio between 3/4 to 4/3.
            This crop is then resized to the given patch size.

        Args:
            patch(tuple): width and height of the patch
            inplace(Boolean): replace the internal images list with the patches
                              if True; otherwise, return the patches.
        '''
        new_imgs = []
        for img in self.imgs:
            area = img.size[0] * img.size[1]
            img_resized = None
            for attempt in range(10):
                target_area = random.uniform(0.08, 1.0) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)
                crop_x = int(round(math.sqrt(target_area * aspect_ratio)))
                crop_y = int(round(math.sqrt(target_area / aspect_ratio)))
                if img.size[0] > crop_x and img.size[1] > crop_y:
                    left_offset = random.randint(0, img.size[0] - crop_x)
                    top_offset = random.randint(0, img.size[1] - crop_y)
                    box = (left_offset, top_offset, left_offset + crop_x,
                           top_offset + crop_y)
                    img_croped = img.crop(box)
                    img_resized = img_croped.resize(patch, Image.BILINEAR)
                    break
            if img_resized is None:
                img_resized = img.resize(patch, Image.BILINEAR)
            new_imgs.append(img_resized)

        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def flip(self, num_case=1, inplace=True):
        '''Randomly flip a img left to right.

        Args:
            num_case: num of cases, must be in {1,2}; if 2, then add the orignal
                      and flipped img
            inplace: inplace imgs or not (return new_imgs)
        '''
        new_imgs = []
        for img in self.imgs:
            if num_case == 1:
                if random.randint(0, 1):
                    new_imgs.append(flip(img))
                else:
                    new_imgs.append(img)
            elif num_case == 2:
                new_imgs.append(flip(img))
                new_imgs.append(img)
            else:
                raise Exception('num_case must be in [0,2]')

        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def flip_down(self, num_case=1, inplace=True):
        '''Randomly flip a img top to bottom.
        Args:
            num_case: num of cases, must be in {1,2}; if 2, then add the orignal
                      and flip_down img
            inplace: inplace imgs or not (return new_imgs)
        '''
        new_imgs = []
        for img in self.imgs:
            if num_case == 1:
                if random.randint(0, 1):
                    new_imgs.append(flip_down(img))
                else:
                    new_imgs.append(img)
            elif num_case == 2:
                new_imgs.append(flip_down(img))
                new_imgs.append(img)
            else:
                raise Exception('num_case must be in [0,2]')

        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def color_cast(self, offset=20, inplace=True):
        '''Add a random value from [-offset, offset] to each channel

        Args:
            offset: cast offset, >0 and <255
            inplace: inplace imgs or not ( return new_imgs)
        '''
        new_imgs = []
        if offset < 0 or offset > 255:
            raise Exception('offset must be >0 and <255')

        for img in self.imgs:
            new_img = color_cast(img, offset)
            new_imgs.append(new_img)
        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs

    def enhance(self, scale=0.2, inplace=True):
        '''Apply random enhancement for Color,Contrast,Brightness,Sharpness.

        Args:
            scale(float): enhancement degree is from [1-scale, 1+scale]
            inplace: inplace imgs or not ( return new_imgs)
        '''
        new_imgs = []
        for img in self.imgs:
            new_img = enhance(img, scale)
            new_imgs.append(new_img)
        if inplace:
            self.imgs = new_imgs
            return self
        else:
            return new_imgs


if __name__ == '__main__':
    tool = ImageTool()
    imgs = tool.load('input.png').\
        resize_by_list([112]).crop5((96, 96), 5).enhance().flip().get()
    for idx, img in enumerate(imgs):
        img.save('%d.png' % idx)
