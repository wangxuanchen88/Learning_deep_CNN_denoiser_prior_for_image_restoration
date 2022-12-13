import os.path
import cv2
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict
import hdf5storage
from scipy import ndimage

import torch


from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util

"""
ref:
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR

"""


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 0.5/255.0  # default: 0, noise level for LR image
    noise_level_model = noise_level_img  # noise level of model, default 0
    model_name = 'ircnn_color'  # 'drunet_gray' | 'drunet_color' | 'ircnn_gray' | 'ircnn_color'
    testset_name = 'set3c'  # test set,  'set5' | 'srbsd68'
    iter_num = 8  # number of iterations
    modelSigma1 = 49
    modelSigma2 = noise_level_model * 255.

    show_img = False  # default: False
    save_L = True  # save LR image
    save_E = True  # save estimated image
    save_LEH = False  # save zoomed LR, E and H images
    border = 0

    # --------------------------------
    # load kernel
    # --------------------------------

    kernels = hdf5storage.loadmat(os.path.join('kernels', 'Levin09.mat'))['kernels']

    sf = 1
    task_current = 'deblur'  # 'deblur' for deblurring
    n_channels = 3 if 'color' in model_name else 1  # fixed
    model_zoo = 'model_zoo'  # fixed
    testsets = './data/test'  # fixed
    results = 'results'  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name)  # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.IRCNN import IRCNN as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
    former_idx = 0

    L_paths = util.get_image_paths(L_path)

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []  # record average PSNR for each kernel

    for k_index in range(kernels.shape[1]):

        test_results = OrderedDict()
        test_results['psnr'] = []
        k = kernels[0, k_index].astype(np.float64)
        util.imshow(k) if show_img else None

        for idx, img in enumerate(L_paths):

            # --------------------------------
            # (1) get img_L
            # --------------------------------

            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)
            img_H = util.modcrop(img_H, 8)  # modcrop

            img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
            util.imshow(img_L) if show_img else None
            img_L = util.uint2single(img_L)

            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN

            # --------------------------------
            # (2) get rhos and sigmas
            # --------------------------------

            rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=iter_num,
                                             modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
            rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

            # --------------------------------
            # (3) initialize x, and pre-calculation
            # --------------------------------

            x = util.single2tensor4(img_L).to(device)

            img_L_tensor, k_tensor = util.single2tensor4(img_L), util.single2tensor4(np.expand_dims(k, 2))
            [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
            FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)

            # --------------------------------
            # (4) main iterations
            # --------------------------------

            for i in range(iter_num):

                # --------------------------------
                # step 1, FFT
                # --------------------------------

                tau = rhos[i].float().repeat(1, 1, 1, 1)
                x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)

                if 'ircnn' in model_name:
                    current_idx = np.int(np.ceil(sigmas[i].cpu().numpy() * 255. / 2.) - 1)

                    if current_idx != former_idx:
                        model.load_state_dict(torch.load(
                            './model_zoo/ircnn_color_image_denoiser_{n}_checkpoint.pth'.format(
                                n=(current_idx + 1) * 2))['model_state_dict'], strict=True)
                        model.eval()
                        for _, v in model.named_parameters():
                            v.requires_grad = False
                        model = model.to(device)
                    former_idx = current_idx

                # --------------------------------
                # step 2, denoiser
                # --------------------------------

                _, x = model(x)

            # --------------------------------
            # (3) img_E
            # --------------------------------

            img_E = util.tensor2uint(x)
            if n_channels == 1:
                img_H = img_H.squeeze()

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name + '_k' + str(k_index) + '_' + model_name + '.png'))

            # --------------------------------
            # (4) img_LEH
            # --------------------------------

            if save_LEH:
                img_L = util.single2uint(img_L)
                k_v = k / np.max(k) * 1.0
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I = cv2.resize(img_L, (sf * img_L.shape[1], sf * img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                util.imshow(np.concatenate([img_I, img_E, img_H], axis=1),
                            title='LR / Recovered / Ground-truth') if show_img else None
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1),
                            os.path.join(E_path, img_name + '_k' + str(k_index) + '_LEH.png'))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name + '_k' + str(k_index) + '_LR.png'))

            psnr = util.calculate_psnr(img_E, img_H, border=border)  # change with your own border
            test_results['psnr'].append(psnr)

        # --------------------------------
        # Average PSNR
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        test_results_ave['psnr'].append(ave_psnr)

        print(test_results['psnr'])
    print(test_results_ave['psnr'])


if __name__ == '__main__':
    main()
