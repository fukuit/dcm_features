''' 形状ファイルを読み込み皮膚表面がフラットになるdicom画像を作成する
'''
import pydicom
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import time


def retrieve_dicom(dicomdir):
    ''' dicom画像ファイルリストを作成する
    '''
    list_dicom = []
    for root_d, _, files in os.walk(dicomdir):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext == '.dcm':
                list_dicom.append(os.path.join(root_d, f))
    return list_dicom


def read_labeldcm(label):
    ''' 形状ファイルを読み込みnumpy array形式の画像データを得る
    '''
    label_dcm = pydicom.dcmread(label)
    if label_dcm[0x0008, 0x0005].value == 'ISO_IR6':
        label_dcm[0x0008, 0x0005].value == 'IOS_IR 6'
    label_dcm.SamplesPerPixel = 1
    label_dcm.ImagePositionPatient = [-75, -25, -75]
    mask = label_dcm.pixel_array.copy()
    return mask


def deformation_dicom(list_dicom, mask, outdir):
    ''' maskの形状に従いlist_dicom中のdicom画像を変形させる
    '''
    ref_dcm = pydicom.dcmread(list_dicom[0])
    depth = len(list_dicom)
    height = ref_dcm.Rows
    width = ref_dcm.Columns
    dcms_array = np.empty((depth, height, width), dtype=ref_dcm.pixel_array.dtype)
    for d in list_dicom:
        dcm = pydicom.dcmread(d)
        idx = list_dicom.index(d)
        array = dcm.pixel_array
        array1 = np.zeros(np.shape(array))
        for w in range(width):
            if np.max(mask[idx, :, w]) != 0:
                maxidx = np.argmax(mask[idx, :, w])
                array1[:, w] = np.concatenate((array[maxidx:, w],
                                               np.zeros(maxidx)))
            else:
                array1[:, w] = np.zeros(height)
        dcm.pixel_array[:, :] = array1.copy()
        dcm.PixelData = dcm.pixel_array.tostring()
        _, filename = os.path.split(d)
        outfile = os.path.join(outdir, filename)
        print(outfile)
        dcm.save_as(outfile)


def main(args):
    ''' main function
    '''
    dicomdir = args[1]
    label = args[2]
    if len(args) == 4:
        outdir = args[3]
    else:
        outdir = '{0}_flatten'.format(args[1])
    if os.path.exists(dicomdir) is True and os.path.exists(label) is True:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        list_dicom = retrieve_dicom(dicomdir)
        mask = read_labeldcm(label)
        deformation_dicom(list_dicom, mask, outdir)


if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) == 4 or len(sys.argv) == 3:
        main(sys.argv)
    else:
        print('{0} <dicom_dir> <label.dcm> <output_dir>'.format(sys.argv[0]))
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
