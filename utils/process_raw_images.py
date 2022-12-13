import cv2
import glob
import numpy as np

"""
This file is used to process the raw images and form the dataset we need. All the raw images will be 
resized to 256x256 and cropped into four none-overlap patches of size 64x64
"""


def process_raw_images(img_paths, output_path,noise_level = 0):
    counter = 0
    for img_path in img_paths:
        img_list = glob.glob(img_path)
        for img_p in img_list:
            img = cv2.imread(img_p).astype(np.int64)

            np.random.seed(0)
            img_noised = img + np.random.normal(0, noise_level, img.shape)
            img_noised[img_noised > 255] = 255
            img_noised[img_noised < 0] = 0

            img = cv2.resize(img.astype(np.uint8), (256, 256))
            img_noised = cv2.resize(img_noised.astype(np.uint8), (256, 256))
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[0:64, 0:64])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[0:64, 0:64])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[0:64, 64:128])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[0:64, 64:128])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[0:64, 128:192])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[0:64, 128:192])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[0:64, 192:])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[0:64, 192:])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[64:128, 0:64])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[64:128, 0:64])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[64:128, 64:128])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[64:128, 64:128])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[64:128, 128:192])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[64:128, 128:192])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[64:128, 192:])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[64:128, 192:])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[128:192, 0:64])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[128:192, 0:64])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[128:192, 64:128])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[128:192, 64:128])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[128:192, 128:192])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[128:192, 128:192])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[128:192, 192:])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[128:192, 192:])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[192:, 0:64])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[192:, 0:64])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[192:, 64:128])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[192:, 64:128])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[192:, 128:192])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[192:, 128:192])
            counter += 1
            cv2.imwrite(output_path + '/original/{c}.jpg'.format(c=counter), img[192:, 192:])
            cv2.imwrite(output_path + '/noised/{c}.jpg'.format(c=counter), img_noised[192:, 192:])
            counter += 1


if __name__ == '__main__':
    # img_paths = ['C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/raw_data/BSD/*',
    #              'C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/raw_data/ImageNet/*',
    #              'C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/raw_data/Waterloo/*'
    #              ]
    # out_put_path = 'C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/data/train'
    img_paths = ['C:/Users/Administrator/Desktop/adcv/Learning deep CNN denoiser prior for image restoration/data/test/set3c/*'
                 ]
    out_put_path = '//data/test/set3c_patches'
    process_raw_images(img_paths, out_put_path)
