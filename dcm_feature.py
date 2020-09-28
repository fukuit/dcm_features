''' DICOMファイルを読み込んで、
表皮1mmを除いた表面から10mm分のMIP画像を作成し、
特徴点を抽出して計数する
'''
import csv
import dicom
import math
import os
import numpy as np
from PIL import Image as image
import cv2
import sys
import time
import json


class config:
    ''' config class
            file: config.json
            method: 画像処理メソッド
    '''
    def __init__(self, file):
        self.file = file
        if os.path.exists(file):
            with open(file, 'r') as f:
                self.json_data = json.load(f)
                self.method = self.json_data['method']
        else:
            dict = {'method': 'AKAZE',
                    'target_all': 'blured_img',
                    'target_all_bin': 'blured_bin_img',
                    'target_mask': 'masked_blured_img',
                    'target_mask_bin': 'masked_blured_bin_img'}
            self.method = 'AKAZE'
            self.json_data = dict
            with open(file, 'w') as f:
                json.dump(dict, f)

    def target_all(self):
        if self.json_data['target_all'] == 'blured_img':
            return self.blured_img
        elif self.json_data['target_all'] == 'bin_img':
            return self.bin_img
        elif self.json_data['target_all'] == 'blured_bin_img':
            return self.blured_bin_img

    def target_all_bin(self):
        if self.json_data['target_all'] == 'bin_img':
            return self.bin_img
        else:
            return self.blured_bin_img

    def target_mask(self):
        if self.json_data['target_mask'] == 'masked_img':
            return self.masked_img
        elif self.json_data['target_mask'] == 'masked_blured_img':
            return self.masked_blured_img
        elif self.json_data['target_mask'] == 'masked_bin_img':
            return self.masked_bin_img
        elif self.json_data['target_mask'] == 'masked_blured_bin_img':
            return self.masked_blured_bin_img

    def target_mask_bin(self):
        if self.json_data['target_mask'] == 'masked_img' or self.json_data['target_mask'] == 'masked_bin_img':
            return self.masked_bin_img
        else:
            return self.masked_blured_bin_img


def make_samplename(dicomlist):
    ''' generate samplename from dicom-filename
    '''
    dcmname = dicomlist[0]
    [fnamebase, _, side, _, _] = os.path.basename(dcmname).split('_')
    samplename = '{0}_{1}'.format(fnamebase, side)
    return samplename


def convert_8bit(img, lower_bound=None, upper_bound=None):
    ''' convert 16bit to 8bit from stackoverflow
    '''
    lower_bound = np.nanmin(img)
    upper_bound = np.nanmax(img)
    lut = np.concatenate([
        np.empty(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


def retrieve_dcms(target):
    ''' retrieve and collect dicom files
    '''
    dicomlist = []
    if os.path.exists(target):
        for root_f, _, files in os.walk(target):
            for dcmfile in files:
                [_, ext] = os.path.splitext(dcmfile)
                if ext == '.dcm':
                    dicomlist.append(os.path.join(root_f, dcmfile))
    else:
        print('{0} not found'.format(target))
    return dicomlist


def savepng(nparr, samplename, outdir, opt):
    ''' save png file
    '''
    if opt == '':
        fname = '{0}.png'.format(samplename)
    else:
        fname = '{0}.{1}.png'.format(samplename, opt)
    im = image.fromarray(nparr)
    pngname = os.path.join(outdir, fname)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    im.save(pngname)


def binarize_image(img):
    ''' binarize
    '''
#    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, th = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th


def create_mask2(img, center=None, radius=None):
    ''' create masked image
    '''
    height, width = img.shape
    if center is None:
        center = [int(width/2), int(height/2)]
    if radius is None:
        if width > height:
            radius = height/4
        else:
            radius = width/4
    mask = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            if math.sqrt((x-center[0])**2+(y-center[1])**2) < radius:
                mask[y, x] = 1
    img = img*mask
    return img


def create_mask(img):
    ''' create masked image
    '''
    height, width = img.shape
    x, y = np.indices((height, width))
    circle = (x - (width/2))**2 + (y - (height/2))**2 < (width/4)**2
    mask = circle.astype(np.uint8)
    img = img*mask
    return img


def mipdicom(dicomlist, samplename, start, thickness):
    ''' generate mip array
    '''
    ref_dcm = dicom.read_file(dicomlist[0])
    [rows, columns] = ref_dcm.pixel_array.shape
    depth = len(dicomlist)
    data = np.empty((8*thickness, depth, columns), dtype=ref_dcm.pixel_array.dtype)
    end = start+8*thickness
    print('{0}:{1}-{2}({3}mm)'.format(samplename, start, end, thickness))
    for dcmf in dicomlist:
        dcmd = dicom.read_file(dcmf)
        tmp = np.empty((rows, columns), dtype=dcmd.pixel_array.dtype)
        tmp[:, :] = dcmd.pixel_array
        data[:, dicomlist.index(dcmf), :] = tmp[start:end, :]
    mipd = np.empty((columns, depth), dtype=data.dtype)
    mipd[:, :] = np.max(data, 0)
    mipd2 = np.flip(mipd, 0)   # 方向転換
    return mipd2


def get_feature(img, samplename, start, thickness, outdir, mask_center=None, radius=None):
    ''' feature detection
    '''
    # create mask parameters
    height, width = img.shape
    if mask_center is None:
        mask_center = [int(width/2), int(height/2)]
    if radius is None:
        if width > height:
            radius = height/4
        else:
            radius = width/4

    # create images
    img8 = convert_8bit(img)
    blured_img = cv2.medianBlur(img8, 5)
#    blured_img = cv2.blur(img, ksize=(10,10))
    bin_img = binarize_image(img8)
    blured_bin_img = binarize_image(blured_img)
    blured_img = blured_img * blured_bin_img
    blured_bin_img = blured_bin_img * 255

    masked_img = create_mask2(img, mask_center, radius)
    masked_img8 = convert_8bit(masked_img)
    masked_blured_img = cv2.medianBlur(masked_img8, 5)
    masked_bin_img = binarize_image(masked_img8)
    masked_blured_bin_img = binarize_image(masked_blured_img)
    masked_blured_img = masked_blured_img * masked_blured_bin_img
    masked_blured_bin_img = masked_blured_bin_img*255

    # output images
    savepng(img8, samplename, outdir, 'orig')
    savepng(blured_img, samplename, outdir, 'blur')
    savepng(masked_img8, samplename, outdir, 'mask')
    savepng(bin_img, samplename, outdir, 'bin')
    savepng(blured_bin_img, samplename, outdir, 'blur_bin')
    savepng(masked_bin_img, samplename, outdir, 'mask_bin')
    savepng(masked_blured_bin_img, samplename, outdir, 'mask_blur_bin')

    # feature detection
    myconf = config('config.json')
    myconf.blured_img = blured_img
    myconf.bin_img = bin_img
    myconf.blured_bin_img = blured_bin_img
    myconf.masked_img = masked_img
    myconf.masked_blured_img = masked_blured_img
    myconf.masked_bin_img = masked_bin_img
    myconf.masked_blured_bin_img = masked_blured_bin_img

    detector = select_detector(myconf.method)
    target_all = myconf.target_all()
    kps_all = detector.detect(target_all)
    response_list = []
    for kp in kps_all:
        response_list.append(kp.response)
    mean_response = np.mean(response_list)
    std_response = np.std(response_list)
    kps_sel_m = []
    kps_sel_ms = []
    for kp in kps_all:
        if kp.response > mean_response:
            kps_sel_m.append(kp)
            if kp.response > mean_response+std_response:
                kps_sel_ms.append(kp)

    target_mask = myconf.target_mask()
    kps_mask = detector.detect(target_mask)
    response_list = []
    for kp in kps_mask:
        response_list.append(kp.response)
    mean_response = np.mean(response_list)
    std_response = np.std(response_list)
    kps_mask_sel_m = []
    kps_mask_sel_ms = []
    for kp in kps_mask:
        if kp.response > mean_response:
            kps_mask_sel_m.append(kp)
            if kp.response > mean_response+std_response:
                kps_mask_sel_ms.append(kp)

    num_kps_sel = len(kps_sel_m)
    num_kps_sel2 = len(kps_sel_ms)
    num_kps = len(kps_all)
    num_bin = np.count_nonzero(myconf.target_all_bin())
    num_mask_kps_sel = len(kps_mask_sel_m)
    num_mask_kps_sel2 = len(kps_mask_sel_ms)
    num_mask_kps = len(kps_mask)
    num_mask_bin = np.count_nonzero(myconf.target_mask_bin())

    output(samplename, mask_center[0], mask_center[1], radius,
           num_kps_sel, num_kps_sel2, num_kps, num_bin,
           num_mask_kps_sel, num_mask_kps_sel2, num_mask_kps, num_mask_bin,
           start, thickness, outdir)

    # output result_image
    # # total image
    out1 = cv2.drawKeypoints(img8, kps_all, None, color=(255, 0, 0), flags=2)
    savepng(out1, samplename, outdir, 'kps')
    out2 = cv2.drawKeypoints(img8, kps_sel_m, None, color=(255, 0, 0), flags=2)
    savepng(out2, samplename, outdir, 'kps_m')
    out3 = cv2.drawKeypoints(img8, kps_sel_ms, None, color=(255, 0, 0), flags=2)
    savepng(out3, samplename, outdir, 'kps_ms')
    # # masked image
    out1 = cv2.drawKeypoints(masked_img8, kps_mask, None, color=(255, 0, 0), flags=2)
    savepng(out1, samplename, outdir, 'mask-kps')
    out2 = cv2.drawKeypoints(masked_img8, kps_mask_sel_m, None, color=(255, 0, 0), flags=2)
    savepng(out2, samplename, outdir, 'mask-kps_m')
    out3 = cv2.drawKeypoints(masked_img8, kps_mask_sel_ms, None, color=(255, 0, 0), flags=2)
    savepng(out3, samplename, outdir, 'mask-kps_ms')

    keypoints = detector.detect(masked_blured_img)
    out = cv2.drawKeypoints(img8, keypoints, None, flags=2)
    savepng(out, samplename, outdir, '0')


def output(samplename, center_x, center_y, radius,
           num_kps_sel, num_kps_sel2, num_kps, num_bin,
           num_mask_kps_sel, num_mask_kps_sel2, num_mask_kps, num_mask_bin,
           start_mm, thickness, outdir):
    ''' output result to csvfile
    '''
    start = int(start_mm/8)
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    score5 = 0
    score6 = 0
    if num_kps_sel != 0 or num_bin != 0:
        score1 = float(num_kps_sel/num_bin)
    if num_kps_sel2 !=0 or num_bin != 0:
        score2 = float(num_kps_sel2/num_bin)
    if num_kps !=0 or num_bin != 0:
        score3 = float(num_kps/num_bin)
    if num_mask_kps_sel != 0 or num_mask_bin != 0:
        score4 = float(num_mask_kps_sel/num_mask_bin)
    if num_mask_kps_sel2 != 0 or num_mask_bin != 0:
        score5 = float(num_mask_kps_sel2/num_mask_bin)
    if num_mask_kps != 0 or num_mask_bin != 0:
        score6 = float(num_mask_kps/num_mask_bin)
    list = [samplename, center_x, center_y, radius,
            num_kps_sel, num_kps_sel2, num_kps, num_bin,
            num_mask_kps_sel, num_mask_kps_sel2, num_mask_kps, num_mask_bin,
            score1, score2, score3, score4, score5, score6]
    fname = 'result.{0}_{1}.csv'.format(start, thickness)
    outputfilepath = os.path.join(outdir, fname)
    if os.path.exists(outputfilepath):
        with open(outputfilepath, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list)
            print(list)
    else:
        with open(outputfilepath, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            title = ['SampleName', 'Center.X', 'Center.Y', 'Radius',
                     'Keypoints(>mean)', 'Keypoints(>mean+sd)', 'Keypoints(all)', 'Vessel Density',
                     'Keypoints(>mean):mask', 'Keypoints(>mean+sd):mask', 'Keypoints(all):mask', 'Vessel Density:mask',
                     'Score1', 'Score2', 'Score3', 'Score4', 'Score5', 'Score6']
            writer.writerow(title)
            writer.writerow(list)
            print(list)


def select_detector(method):
    ''' select CV2 detector '''
    if method == 'FAST':
        detector = cv2.FAST_create()
    elif method == 'MSER':
        detector = cv2.MSER_create()
    elif method == 'ORB':
        detector = cv2.ORB_create()
    elif method == 'BRISK':
        detector = cv2.BRISK_create()
    elif method == 'KAZE':
        detector = cv2.KAZE_create()
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create(descriptor_type=5)
    elif method == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()
    return detector


def main(target, outdir, start, thickness, mask_center=None, radius=None):
    ''' main
    '''
    dicomlist = retrieve_dcms(target)
    if len(dicomlist) == 0:
        return
    samplename = make_samplename(dicomlist)
    print(samplename)
    outdir = os.path.join(outdir, '{0}_{1}'.format(start, thickness))
    start_mm = int(start*8)
    mip = mipdicom(dicomlist, samplename, start_mm, thickness)
    get_feature(mip, samplename, start_mm, thickness, outdir, mask_center, radius)


if __name__ == '__main__':
    start_time = time.time()


    if len(sys.argv) > 4:
        target = sys.argv[1]
        if not os.path.exists(target):
            exit
        outdir = sys.argv[2]
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        start = float(sys.argv[3])
        thickness = int(sys.argv[4])

        if len(sys.argv) == 8:
            mask_center = [float(sys.argv[5]), float(sys.argv[6])]
            radius = float(sys.argv[7])
        else:
            mask_center = None
            radius = None
        main(target, outdir, start, thickness, mask_center, radius)
    else:
        print('{0} targetdir out_dir start thickness center.x center.y radius'.format(sys.argv[0]))

    elapsed_time = time.time() - start_time
    print('elapsed_time:{0} [sec]'.format(elapsed_time))
'''
end
'''
