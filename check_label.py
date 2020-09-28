'''
    upperSurfaceLabel.dcmファイルを一覧で表示させる（チェックのため）
'''
import dicom
import os
import sys
from matplotlib import pyplot as plt


def plot_function_name(name, x=0, y=0):
    '''
    pyplotのグラフの中央部にグラフのタイトルを表示する
    '''
    plt.text(x, y, name, alpha=1, size=10, ha="center", va="center")


def main(target_dir, mode):
    '''
    upperSufraceLabel.dcmを探索し、axialで100枚目の画像を並べて表示する
    '''
    print(target_dir)
    print(os.path.exists(target_dir))
    dcmlist = []
    for root_dir, _, files in os.walk(target_dir):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext == '.dcm':
                dcmlist.append(os.path.join(root_dir, f))
    print(len(dcmlist))
    i = 1
    for f in dcmlist:
        d = dicom.read_file(f)
        title = os.path.basename(f).replace('_upperSurfaceLabel.dcm', '')
        if mode == 'sugittal':
            m = d.pixel_array[:, :, 100]
            plt.subplot(6, 8, i)
            xpos = int(m.shape[1]/2)
            ypos = int(m.shape[0]/2)
            plt.imshow(m)
            plot_function_name(title, x=xpos, y=ypos)
        elif mode == 'axial':
            m = d.pixel_array[100, :, :]
            plt.subplot(8, 6, i)
            xpos = int(m.shape[1]/2)
            ypos = int(m.shape[0]/2)
            plt.imshow(m)
            plot_function_name(title, x=xpos, y=ypos)
        elif mode == 'coronal':
            m = d.pixel_array[:, 250, :]
            plt.subplot(7, 7, i)
            xpos = int(m.shape[1]/2)
            ypos = int(m.shape[0]/2)
            plt.imshow(m)
            plot_function_name(title, x=xpos, y=ypos)
        i += 1
    plt.show()

if __name__ == '__main__':
    modes = ['sugittal', 'axial', 'coronal']
    args = sys.argv
    if len(args) != 2:
        mode = modes[1]
    else:
        mode = args[1]
    main('.\LABELDIR', mode)
