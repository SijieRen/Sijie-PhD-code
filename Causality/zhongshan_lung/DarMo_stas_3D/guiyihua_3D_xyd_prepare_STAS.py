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

window_center = -600
window_width = 1200

# sitk_image = sitk.GetImageFromArray(image_array)
# sitk.WriteImage(itk_image, "output.nii.gz") 

def adjustGrayImage_stdMean(img):
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

# 中山医院提供的resample code
# def resample(image, spacing, new_spacing=[0.5, 0.5, 0.5], order=1):  # [0.5, 0.5, 0.5]
#     # shape can only be int, so has to be rounded.
#     new_shape = np.round(image.shape * spacing / new_spacing)
#     # the actual spacing to resample.
#     resample_spacing = spacing * image.shape / new_shape
#     resize_factor = new_shape / image.shape
#     image_new = scipy.ndimage.interpolation.zoom(image, resize_factor, mode='nearest', order=order)
#     # if image.shape[1] == 1024:
#     #     image_new = scipy.ndimage.interpolation.zoom(image_new, [1, 0.5, 0.5], mode='nearest', order=order)
#     return image_new, resample_spacing, resize_factor

def get_basic_inf(path, pathofmask, if_resample=False):
    # 读取数据
    image = sitk.ReadImage(path)
    image_mask = sitk.ReadImage(pathofmask)
    image_spacing = image.GetSpacing()
    mask_spacing = image_mask.GetSpacing()

    # std mean 统一归一化数据，不使用窗宽窗位进行调整
    image = adjustGrayImage_stdMean(sitk.GetArrayFromImage(image))
    

    # convert to ndarry# 输出数据shape
    # 将CT矩阵从[Z,Y,X]换至[X,Y,Z],并读取各自spacing (TT:未换，为了适应后面resample中[Z,Y,X]的顺序)
    arrayofCT = np.array(image)  # 使用GetArrayFromImage时，X,Z位置调换，输出形状为：(Depth, Height, Width)即（Z,Y,X）。
    # spacing = np.array(image.GetSpacing())
    # spacing = [spacing[2], spacing[1], spacing[0]]
    # spacing = np.array(spacing)
    # 将结节矩阵从[Z,Y,X]换至[X,Y,Z]
    arrayofMask = np.array(sitk.GetArrayFromImage(image_mask))
    # spacing1 = np.array(image.GetSpacing())
    # spacing1 = [spacing1[2], spacing1[1], spacing1[0]]
    # spacing1 = np.array(spacing1)

    # 取出非零部分（即结节）
    arrayofLNLo = np.array(np.nonzero(arrayofMask))  # 返回非零元素的位置,返回矩阵形状为3*（非零元素个数），每行分别代表一个维度的坐标
    # print("shape.", arrayofLNLo.shape)
    # print(np.min(arrayofLNLo, axis=0), np.max(arrayofLNLo, axis=0))
    print(np.min(arrayofLNLo, axis=1), np.max(arrayofLNLo, axis=1))
    ZYX_chazhi = np.max(arrayofLNLo, axis=1) - np.min(arrayofLNLo, axis=1)
    print("ZYX_chazhi: ", ZYX_chazhi) # Z Y X
    # print(np.min(arrayofLNLo, axis=2), np.max(arrayofLNLo, axis=2))
    ZCo = list(arrayofLNLo[0])
    YCo = list(arrayofLNLo[1])
    XCo = list(arrayofLNLo[2])
    # print("X array, Y array, Z array",XCo,YCo,ZCo)
    arrayofLNCoor = [XCo, YCo, ZCo]
    # arrayofLNCoor = np.array(arrayofLNCoor).T  # 将非零元素位置转换为标准的三维坐标形式 ZYX
    # print('the shape of nodule array is %d rows and %d columns' % (arrayofLNCoor.shape[0], arrayofLNCoor.shape[1]))
    # 取所有结节的中心点作为结节中心
    centerZYX = ((np.min(arrayofLNLo, axis=1) + np.max(arrayofLNLo, axis=1))/2).astype(int)  # 返回整个结节的中心点
    print("center coordinate", centerZYX)
    # print('the center of nodule(calculated using all points) is:%s' % center)
    return arrayofCT, arrayofMask, centerZYX, ZYX_chazhi, image_spacing, mask_spacing


def img_display_and_save(arrayofCT, centerZYX, count_saveimg, save_path, ZYX_chazhi, image_spacing, mask_spacing):

    sizeofcuts = []
    for ii in range(len(ZYX_chazhi)):
        if ZYX_chazhi[ii] % 2 == 0:
            sizeofcuts.append(ZYX_chazhi[ii])
        else:
            sizeofcuts.append(ZYX_chazhi[ii]+1)

    # 横截面
    # 此处提取64*64*64 cube
    max_mask_coordinate = np.max(ZYX_chazhi)
    if max_mask_coordinate %2 ==0:
        max_mask_coordinate = max_mask_coordinate
        print("%2 = 0")
    else:
        max_mask_coordinate += 1
        print("%2 != 0")
    
    print("max ground: ",max_mask_coordinate, "of ", sizeofcuts)

    background2 = np.zeros((max_mask_coordinate, max_mask_coordinate, max_mask_coordinate), dtype=np.uint8)  # 用作截取结节图像的背景 Z Y X
    img = arrayofCT

    Z_0 = int(centerZYX[0] - sizeofcuts[0]/ 2)  
    Z_1 = int(centerZYX[0] + sizeofcuts[0]/ 2)
    Y_0 = int(centerZYX[1] - sizeofcuts[1] / 2)
    Y_1 = int(centerZYX[1] + sizeofcuts[1] / 2)
    X_0 = int(centerZYX[2] - sizeofcuts[2] / 2)
    X_1 = int(centerZYX[2] + sizeofcuts[2] / 2)
    if Z_0 < 0:
        Z_0 = 0
    if Y_0 < 0:
        Y_0 = 0
    if X_0 < 0:
        X_0 = 0
    if Z_1 > arrayofCT.shape[0]:  # 数据集中有部分数据只有40-60个slice，在裁剪时row会超出上下界
        Z_1 = arrayofCT.shape[0]
    if Y_1 > arrayofCT.shape[1]:
        Y_1 = arrayofCT.shape[1]
    if X_1 > arrayofCT.shape[2]:
        X_1 = arrayofCT.shape[2]
    cubecut = img[Z_0:Z_1, Y_0:Y_1, X_0: X_1]# Z Y X  # 注意参数顺序，前面的表示行，后面的表示列,高，对应的是坐标位置，不要和坐标值搞混淆；且使用：时注意，1：n表示从1到n，不包括n
    # meanofimgcut2 = int(np.array(cubecut).mean())  # 对截取图像像素取平均
    # background2[:, :, :] = meanofimgcut2 # padding均值灰度  #注释掉 padding0
    # print((max_mask_coordinate - sizeofcuts[0])/2, (max_mask_coordinate - sizeofcuts[0])/2)
    # print((max_mask_coordinate - sizeofcuts[1])/2, (max_mask_coordinate - sizeofcuts[1])/2)
    # print((max_mask_coordinate - sizeofcuts[2])/2, (max_mask_coordinate - sizeofcuts[2])/2)
    background2[int((max_mask_coordinate - sizeofcuts[0])/2): int((max_mask_coordinate + sizeofcuts[0])/2), 
                int((max_mask_coordinate - sizeofcuts[1])/2): int((max_mask_coordinate + sizeofcuts[1])/2), 
                int((max_mask_coordinate - sizeofcuts[2])/2): int((max_mask_coordinate + sizeofcuts[2])/2)] \
                    = np.uint8(cubecut[:, :, :])  # 用所截取图像来填充背景，若截取图像小于背景图大小，对于空出部分，已用平均灰度/0填充
    # -check2 把剪切图像放在中心、

    # np.save(path + str(count_saveimg) + '.npy', background2)

    # 楚然推荐的resample方法：train：保持原样; test：spacing(1, 1, 1)
    # image_bg2 = sitk.GetImageFromArray(background2)
    # image_bg2_resample = resample_image(image_bg2, image_spacing, "image", target_spacing=(1, 1, 1))
    # background2 = sitk.GetArrayFromImage(image_bg2_resample)

    # print("cutcude:", cubecut.shape, "bg2", background2.shape)
    np.save(save_path, background2)
    print("not saved !!!!!!!!!!!!!")
    print(f'{count_saveimg} saved---------')

def read_nii_file(nii_file_path):
    sitk_data = sitk.ReadImage(nii_file_path)
    return sitk.GetArrayFromImage(sitk_data), sitk_data 


def prepare_orimage(data_root_path, output_dir, if_save=False): 
    print("save in :", output_dir) 
    for type_name in os.listdir(data_root_path):
        type_name_path = os.path.join(data_root_path, type_name)
        output_path = os.path.join(output_dir, type_name)
        os.makedirs(output_path, exist_ok=True)
        origin_case_name_list = [case_name for case_name in os.listdir(type_name_path) if#中山医院_mask 肿瘤医院_seg
                                 "_mask.nii.gz" not in case_name] #TODO 对不同数据是否需要修改处理后缀
        mask_case_name_list = [case_name for case_name in os.listdir(type_name_path) if #中山医院_mask 肿瘤医院_seg
                                 "_mask.nii.gz" in case_name] #TODO 对不同数据是否需要修改处理后缀
        save_p = 0
        for origin_case_name in origin_case_name_list:
            origin_case_name_path = os.path.join(type_name_path, origin_case_name)
            # seg_case_name_path = os.path.join(type_name_path, origin_case_name.replace(".nii.gz", "_seg.nii.gz"))#for 肿瘤医院
            
            for mask_name in mask_case_name_list:
                if origin_case_name[:-7] in mask_name: # XXX.nii.gz for origin_case_name
                    mask_case_name = mask_name
                    # break
                    seg_case_name_path = os.path.join(type_name_path, mask_case_name)# for 中山医院 and 肿瘤医院
                    print("99999999",seg_case_name_path)
                    mask_image, _ = read_nii_file(seg_case_name_path)  
                    # arrayofCT, arrayofMask, centerXYZ = get_basic_inf(CT_file_path, mask_file_path)
                    # img_display_and_save(arrayofCT, centerXYZ, key_id[index], path_savecuts)

                    arrayofCT, arrayofMask, centerZYX, max_min, image_spacing, mask_spacing = get_basic_inf(origin_case_name_path, seg_case_name_path)
                    origin_prefix_name = origin_case_name[:-7]##中山[:-7]  肿瘤[:-7]
                    output_image_dir = os.path.join(output_path, origin_prefix_name)
                    #print("origin_prefix_name",origin_prefix_name)
                    os.makedirs(output_image_dir, exist_ok=True)
                    output_save_name = f"{str(save_p + 1).zfill(3)}.npy" #同一个人如果有多张图片是否在不同depth
                    origin_output_image_name = os.path.join(output_image_dir, output_save_name)
                    #print("output_image_dir", output_image_dir)
                    #print("SAVE SAVE", origin_output_image_name)
                    # origin_output_image.save(origin_output_image_name)

                    if if_save:
                        img_display_and_save(arrayofCT, centerZYX, save_p, origin_output_image_name, max_min, image_spacing, mask_spacing)
                        print("save num:", save_p)
                    else:
                        print("do not save {}, only print".format(save_p))

                    save_p+=1


    return None


if __name__ == '__main__':
    data_root_path = "../../../../data/zhongshan_Lung_data/中山医院/" 
    output_dir = "../../../../data/Total_Stas_unzip_zhongshan_Lung_data_3D-bgMAX_maskXYZgezi_padding0_stdMean_noResample-check2/中山医院/"
    if_save = 1
    prepare_orimage(data_root_path, output_dir, if_save)

