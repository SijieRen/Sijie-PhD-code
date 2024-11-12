import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
# 生成实验所用的32*32结节图片npy数据
import numpy as np
from os import path
import pandas as pd
import os
from matplotlib import pyplot as plt
from PIL import Image
from collections import Counter
import scipy
from scipy import ndimage
from glob import glob

import SimpleITK as sitk

self.Img=sitk.ReadImage(self.ct_path)
ImgArr = sitk.GetArrayFromImage(self.Img)
ImgArr = self.adjustGrayImage_stdMean(ImgArr)
ImgArr_resampled = sitk.GetArrayFromImage(self.ResampledImg[self.needindex.index(i)])
ImgArr_resampled = self.adjustGrayImage_stdMean(ImgArr_resampled)

def adjustGrayImage_stdMean(self, img):
        mean = np.mean(img)
        std = np.std(img)
        eps = 1e-6
        return (img - mean) / (std + eps)

def resample_image(image, original_spacing, mode, target_spacing=(1, 1, 1)):
    if original_spacing is None:
        ## 有部分数据是nrrd格式，没有original_spacing
        ## 直接返回原图，就不调整spacing了
        return image
    if original_spacing[2] > 10:
        return image
    # image: a SimpleITK image object
    # target spacing: a list or tuple of three numbers
    # return: a resampled SimpleITK image object

    # get the original spacing, direction and origin of the image
    # original_spacing = image.GetSpacing()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()
    # get the original size of the image
    original_size = list(image.GetSize())

    # calculate the new size of the image based on the target spacing and original spacing
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in
                zip(original_size, original_spacing, target_spacing)]

    if original_size == new_size:
        return image

    # create a resample filter with linear interpolation
    resample_filter = sitk.ResampleImageFilter()
    if mode == 'image':
        resample_filter.SetInterpolator(sitk.sitkLinear)
    else:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    # set the output spacing, direction, origin and size
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetOutputDirection(original_direction)
    resample_filter.SetOutputOrigin(original_origin)
    resample_filter.SetSize(new_size)

    # execute the resampling on the input image
    resampled_image = resample_filter.Execute(image)

    return resampled_image