import os
import SimpleITK as sitk
import numpy as np
from PIL import Image


window_center = -600
window_width = 1200


def get_npy_path(data_root_path):  
    for sample_name in os.listdir(data_root_path):
        sample_path = os.path.join(data_root_path, sample_name)
        for png_name in os.listdir(sample_path):
            png_path = os.path.join(sample_path, png_name)
            print(png_path)

    return None


if __name__ == '__main__':
    #data_root_path = "../../../../data/zhongshan_Lung_data/中山医院/阴性/" 
    output_dir = "../../../../data/Total_Stas_unzip_zhongshan_Lung_data_3D/中山医院/阳性/"
    get_npy_path(output_dir)

