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

import random
import numpy as np
from PIL import Image,ImageEnhance

def load_img(path, grayscale=False):
  img = Image.open(path)
  if grayscale:
      img = img.convert('L')
  else:  # Ensure 3 channel even when loaded image is grayscale
      img = img.convert('RGB')
  return img

def do_crop(img,crop,position):

    if img.size[0] < crop[0]:
        raise Exception('img size[0] %d is smaller than crop[0]: %d' % (img[0],crop[0]))
    if img.size[1] < crop[1]:
        raise Exception('img size[1] %d is smaller than crop[1]: %d' % (img[1],crop[1]))

    if position == 'left_top':
        left,upper=0,0
    elif position == 'left_bottom':
        left,upper=0,img.size[1]-crop[1] 
    elif position == 'right_top':
        left,upper=img.size[0]-crop[0],0
    elif position == 'right_bottom':
        left,upper=img.size[0]-crop[0],img.size[1]-crop[1] 
    elif position == 'center':
        left,upper=(img.size[0]-crop[0])/2,(img.size[1]-crop[1])/2
    else:
        raise Exception('position is wrong')

    box =(left,upper,left+crop[0],upper+crop[1]) 
    new_img = img.crop(box)
    #print "crop to box %d,%d,%d,%d" % box
    return new_img

def do_crop_and_scale(img,crop,position):
    size = img.size
    if position == 'left':
        left,upper=0,0
        right,bottom = size[1],size[1]
    elif position == 'center':
        left,upper=(size[0]-size[1])/2,0 
        right,bottom =(size[0]+size[1])/2,size[1]
    elif position == 'right':
        left,upper=size[0]-size[1],0 
        right,bottom =size[0],size[1]
    elif position == 'top':
        left,upper=0,0 
        right,bottom =size[0],size[0]
    elif position == 'middle':
        left,upper=0,(size[1]-size[0])/2
        right,bottom =size[0],(size[1]+size[0])/2
    elif position == 'bottom':
        left,upper=0,size[1]-size[0]
        right,bottom =size[0],size[1]
    else:
        raise Exception('position is wrong')
    box =(left,upper,right,bottom) 
    new_img = img.crop(box)

    new_img = img.resize(crop)
    #print box+crop
    #print "crop to box %d,%d,%d,%d and scale to %d,%d" % (box+crop)
    return new_img

def do_resize(img,small_size):
    size = img.size
    if size[0]<size[1]:
        new_size = ( small_size, int(small_size*size[1]/size[0]) )
    else:
        new_size = ( int(small_size*size[0]/size[1]), small_size )
    new_img=img.resize(new_size)
    #print 'resize to (%d,%d)' % new_size
    return new_img
 

def do_color_cast(img,offset):

    x = np.asarray(img, dtype='uint8')
    x.flags.writeable = True  
    cast_value=[0,0,0]
    for i in range(3):
        r= random.randint(0,1)
        if r:
            cast_value[i] = random.randint(-offset,offset)
    for w in range(img.size[0]):
        for h in range(img.size[1]):
            for c in range(3):
                if cast_value[c]==0:
                    continue
                v=x[w][h][c]+cast_value[c] 
                if v<0:
                    v=0
                if v>255:
                    v=255
                x[w][h][c]=v
    new_img= Image.fromarray(x.astype('uint8'), 'RGB')
    return new_img


def do_enhance(img,scale):

    # Color,Contrast,Brightness,Sharpness
    enhance_value=[1.0,1.0,1.0,1.0]
    for i in range(4):
        r= random.randint(0,1)
        if r:
            enhance_value[i] = random.uniform(1-scale,1+scale)
    if not enhance_value[0]==1.0:
        enhancer = ImageEnhance.Color(img) 
        img = enhancer.enhance(enhance_value[0])
    if not enhance_value[1]==1.0:
        enhancer = ImageEnhance.Contrast(img) 
        img = enhancer.enhance(enhance_value[1])
    if not enhance_value[2]==1.0:
        enhancer = ImageEnhance.Brightness(img) 
        img = enhancer.enhance(enhance_value[2])
    if not enhance_value[3]==1.0:
        enhancer = ImageEnhance.Sharpness(img) 
        img = enhancer.enhance(enhance_value[3])
    return img

def do_flip(img):
    #print 'flip'
    new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return new_img     
 
def get_list_sample(l,sample_size):
    return [ l[i] for i in sorted(random.sample(xrange(len(l)), sample_size))]

class Imgtool():

    def __init__(self):
        self.imgs=[]
        return

    def load(self,path):
        img = load_img(path)
        self.imgs=[img]
        return self

    def set(self,imgs):
        self.imgs = imgs
        return self

    def append(self,img):
        self.imgs.append(img)
        return self

    def get(self):
        return self.imgs
 
    def resize_by_range(self,rng,k=1,update=True):

        '''
        Args:
            rng: a tuple (begin,end), include begin, exclude end
            k: number of samples, must be smaller than or equare to the length of 
               the range list, if k=0, then sample all
            update: update imgs or not ( return new_imgs)
        '''
        size_list = range(rng[0],rng[1])
        return self.resize_by_list(size_list,k,update)
        
    def resize_by_list(self,size_list,k=1,update=True):
        '''
        Args:
            k: number of samples, must be smaller than or equare to the length of 
               size_list, if k=0, then sample all
            update: update imgs or not ( return new_imgs)
        '''
        new_imgs=[]
        if k<0 or k > len(size_list):
            raise Exception('k must be smaller in [0,%d(length of size_list)]' % len(size_list))
        for img in self.imgs:
            if k ==0 or k== len(size_list):
                small_sizes = size_list
            else:
                small_sizes = get_list_sample(size_list,k) 

            for small_size in small_sizes:
                new_img= do_resize(img,small_size)
                new_imgs.append(new_img)
        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs

    def resize_for_test(self,rng):
        '''
        Args:
            rng: a tuple (begin,end)
        '''
        size_list=[rng[0],rng[0]/2+rng[1]/2,rng[1]]
        return self.resize_by_list(size_list,k=3)

    def rotate_by_range(self,rng,k=1,update=True):

        '''
        Args:
            rng: a tuple (begin,end), include begin, exclude end
            k: number of samples, must be smaller than or equare to the length of 
               the range list, if k=0, then sample all
            update: update imgs or not ( return new_imgs)
        '''
        size_list = range(rng[0],rng[1])
        return self.rotate_by_list(angle_list,k,update)
        
    def rotate_by_list(self,angle_list,k=1,update=True):
        '''
        Args:
            k: number of samples, must be smaller than or equare to the length of 
               size_list, if k=0, then sample all
            update: update imgs or not ( return new_imgs)
        '''
        new_imgs=[]
        if k<0 or k > len(angle_list):
            raise Exception('k must be smaller in [0,%d(length of angle_list)]' % len(angle_list))

        for img in self.imgs:
            if k ==0 or k== len(angle_list):
                angles = angle_list
            else:
                angles = get_list_sample(angle_list,k) 

            for angle in angles:
                new_img= img.rotate(angle)
                new_imgs.append(new_img)
        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs


    def crop_5(self,crop_size,k=1,update=True):
        '''
        Args:
            k: number of samples, must be in [0,5], if k=0, then sample all
            update: update imgs or not ( return new_imgs)
        '''
        new_imgs=[]
        positions=["left_top","left_bottom","right_top","right_bottom","center"]
        if k > 5 or k <0:
            raise Exception('k must be in [0,5]')
        for img in self.imgs:

            if k>0 and k<5:
                positions = get_list_sample(positions,k) 

            for position in positions:
                new_img=do_crop(img,crop_size,position)
                new_imgs.append(new_img)

        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs

    def crop_and_scale(self,crop_size,k=1,update=True):
        '''
        crop and scale the image into the given size.
        According to img size, crop position could be either (left, center, right)
        or (up, center, low).

        Args:
            k: number of samples, must be in [0,3], if k=0, then sample all
            update: update imgs or not ( return new_imgs)
        '''
        if not crop_size[0] == crop_size[1]:
            raise Exception('crop_size must be a square')
        new_imgs=[]
        if k > 3 or k <0:
            raise Exception('k must be in [0,3]')
        positions_horizental=["left","center","right"]
        positions_vertical=["top","middle","bottom"]
        for img in self.imgs:
            size = img.size
            if size[0] > size[1]: 
                if k>0 and k<3:
                    positions = get_list_sample(positions_horizental,k) 
                else:
                    positions = positions_horizental 
            else:
                if k>0 and k<3:
                    positions = get_list_sample(positions_vertical,k) 
                else:
                    positions = positions_vertical 
 
            for position in positions:
                new_img=do_crop_and_scale(img,crop_size,position)
                new_imgs.append(new_img)

        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs

    def crop_union(self,crop_size,k=1,update=True):

        '''
        this is a union of crop_5 and crop_and_scale 
        you can follow this example to union any number of imgtool methods
        '''
        crop_5_num = 5
        crop_and_scale_num = 3
        if k<0 or k> crop_5_num+crop_and_scale_num:
            raise Exception('k must be in [0,%d]' % (crop_5_num+crop_and_scale_num) )
        if k==0 or k == crop_5_num+crop_and_scale_num:
            count = crop_5_num
        else:
            sample_list = range(0,crop_5_num+crop_and_scale_num)
            samples = get_list_sample(sample_list,k)
            count=0
            for s in samples:
                if s < crop_5_num:
                    count+=1 
        new_imgs=[]
        if count > 0:
            new_imgs += self.crop_5(crop_size,k=count,update=False) 
        if k-count > 0:
            new_imgs += self.crop_and_scale(crop_size,k=k-count,update=False)

        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs
  
    def flip(self,k=1,update=True):
        '''
        randomly flip a img left to right
        Args:
            k: number of samples, must be in [0,1,2], if k=0,2, then sample all
            update: update imgs or not ( return new_imgs)
        '''
        new_imgs=[]
        if k<0 or k>2:
            raise Exception('k must be in [0,2]')
        for img in self.imgs:
            flips = [0,1]
            if k>0 and k <2:
                r=random.randint(0,1)
                flips = [r]
            for flip in flips:
                if flip: 
                    new_img=do_flip(img) 
                else:
                    new_img=img              
                new_imgs.append(new_img)

        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs

    def color_cast(self,offset=20,k=1,update=True):
        '''
        randomly do color cast on rgb channels of a img
        Args:
            offset: cast offset, >0 and <255
            k: number of samples, must be larger than 0 
            update: update imgs or not ( return new_imgs)
        '''
        new_imgs=[]
        if k<=0:
            raise Exception('k must be larger than 0')
        if offset<0 or offset>255:
            raise Exception('offset must be >0 and <255')
         
        for img in self.imgs:
            for i in range(k):
                new_img=do_color_cast(img,offset)
                new_imgs.append(new_img)
        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs

    def enhance(self,scale=0.2,k=1,update=True):
        '''
        randomly do color, contrast, brightness and sharpness enhance on img
        Args:
            scale: cast scale, >0 and <1
            k: number of samples, must be larger than 0 
            update: update imgs or not ( return new_imgs)
        '''
        new_imgs=[]
        if k<=0:
            raise Exception('k must be larger than 0')
        for img in self.imgs:
            for i in range(k):
                new_img=do_enhance(img,scale)
                new_imgs.append(new_img)
        if update:
            self.imgs=new_imgs
            return self
        else:
            return new_imgs

