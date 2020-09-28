# -*- coding: utf-8 -*-
'''Created: 2017/04/21
@author: FUKUI Toshifumi <fukuit.toshifumi@canon.co.jp>

DICOM画像のaxial面のslice送り方向を反転させ、保存する

更新履歴:
  2017/04/25    Version: 1.1 オリジナルのファイルには、ファイル名に.origをつけてバックアップする
  2017/04/21    Version: 1.0
'''

import dicom
import os
import shutil
import sys
import numpy as np
from matplotlib import pyplot as plt


def show_img(arr1, arr2, depth, row, col):
    ''' 変形前後の画像の代表的なところを適当に表示する
    '''
    plt.subplot(2, 3, 1)
    plt.imshow(arr1[depth, :, :])
    plt.title('before: depth={0}'.format(depth))
    plt.subplot(2, 3, 2)
    plt.imshow(arr1[:, row, :])
    plt.title('before: row={0}'.format(row))
    plt.subplot(2, 3, 3)
    plt.imshow(arr1[:, :, col])
    plt.title('before: col={0}'.format(col))
    plt.subplot(2, 3, 4)
    plt.imshow(arr2[depth, :, :])
    plt.title('after: depth={0}'.format(depth))
    plt.subplot(2, 3, 5)
    plt.imshow(arr2[:, row, :])
    plt.title('after: row={0}'.format(row))
    plt.subplot(2, 3, 6)
    plt.imshow(arr2[:, :, col])
    plt.title('after: col={0}'.format(col))
    plt.show()


def make_outfilepath(labelfile):
    ''' labelファイルから、新しいファイル名を生成する
    '''
    basename, ext = os.path.splitext(labelfile)
    outfile = '{0}.orig{1}'.format(basename, ext)
    return outfile


def backup_orig(labelfile):
    ''' labelfileのファイル名に.origをつけてバックアップとしてコピーする
    '''
    backupfile = make_outfilepath(labelfile)
    shutil.copy2(labelfile, backupfile)


def reverse_label(labelfile, check):
    ''' labelファイルを上下反転させる
    '''
    label = dicom.read_file(labelfile)
    # DICOMヘッダの不備を修正する
    if label[0x0008, 0x0005].value == 'ISO_IR6':
        label[0x0008, 0x0005].value = 'ISO_IR 6'
    label.SamplesPerPixel = 1
    label.ImagePositionPatient = [-75, -25, -75]
    # 画像データを取得する
    arr = label.pixel_array.copy()
    # 反転する
    reverse_arr = np.flipud(arr)
    # 反転画像データを保存する
    label.pixel_array = reverse_arr.copy()
    label.PixelData = reverse_arr.tostring()
    # original dataをコピーしてバックアップ
    backup_orig(labelfile)
    label.save_as(labelfile)

    # before / afterを表示(結果チェック用)
    if check is True:
        show_img(arr, reverse_arr, 250, 200, 200)


def main(args):
    ''' main function
    '''
    labelfile = args[1]
    if os.path.exists(labelfile):
        print('target:\t{0}'.format(labelfile))
        # 結果表示が不要な場合はcheck=Falseにする
        check = True
        reverse_label(labelfile, check)

if __name__ == '__main__':
    version = '1.1'
    if len(sys.argv) == 2:
        main(sys.argv)
    else:
        print('Version: {0}'.format(version))
