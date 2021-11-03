from imagenet_c import corrupt
import dataset.CMU as CMU
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
import os
from PIL import Image

# save_path = '/data/input/datasets/VL-CMU-CD/struc_test/corrupted_test'
save_path = '/data/input/datasets/VL-CMU-CD/struc_test/corrupted_test_DRtamnet'

VAL_DATA_PATH = '/data/input/datasets/VL-CMU-CD/'



test_dataset = CMU.Dataset(VAL_DATA_PATH, 'val', transform=True, transform_med=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)


    #
    # :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    # :param severity: strength with which to corrupt x; an integer in [0, 5]
    # :param corruption_name: specifies which corruption function to call;
    # must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    #                 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    #                 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    #                 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
    #                 the last four are validation functions
    # :param corruption_number: the position of the corruption_name in the above list;
    # an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    # :return: the image x corrupted by a corruption function at the given severity; same shape as input
    # path_to_save_t1 = os.path.join(save_path,type,'Image_T1',folderno[1])

corrupt_type = [ 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur','glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'fog',
                     'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
severity = ['1','2','3','4','5']
for type in corrupt_type:
    for num in severity:
        print(type, num)
        for i, batch in enumerate(test_loader):
            t1,t2,label,n1,n2, maskname = batch
            n1 = ''.join(n1)
            n2 = ''.join(n2)
            maskname = ''.join(maskname)


            t1 = (t1.numpy()).squeeze(0).reshape(224, 224, 3)
            t2 = (t2.numpy()).squeeze(0).reshape(224,224,3)
            label = (label.numpy()).squeeze(0)

            Path(os.path.join(save_path,type,num)).mkdir(parents=True, exist_ok=True)
            # t1 = np.transpose(t1, (1, 2, 0))
            imgname = os.path.split(n1)

            imgname1 = os.path.split(n2)
            m1 = os.path.split(maskname)


            folderno = os.path.split(imgname[0])
            imgname = os.path.splitext(imgname[1])
            imgname1 = os.path.splitext(imgname1[1])
            m1 = os.path.splitext(m1[1])


            # t0_image = os.path.split(folderno[0])
            # imgname1 = os.path.split(str(img2_path))
            path_to_save_t0 = os.path.join(save_path,type,num,'raw',folderno[1],'RGB')
            path_to_save_t1 = os.path.join(save_path,type,num,'raw',folderno[1],'RGB')
            path_to_save_label = os.path.join(save_path,type,num,'raw',folderno[1],'GT')

            Path(path_to_save_t0).mkdir(parents=True, exist_ok=True)
            Path(path_to_save_t1).mkdir(parents=True, exist_ok=True)
            Path(path_to_save_label).mkdir(parents=True, exist_ok=True)

            corrupt_image1 = corrupt(t1, corruption_name=type,severity=int(num))
            corrupt_image2 = corrupt(t2, corruption_name=type, severity=int(num))
            corrupt_image1 = Image.fromarray(corrupt_image1)
            corrupt_image2 = Image.fromarray(corrupt_image2)
            # print(path_to_save_label,m1[0])
            label = Image.fromarray(label)

            corrupt_image1 = corrupt_image1.resize((512, 512), Image.ANTIALIAS)
            corrupt_image2 = corrupt_image2.resize((512, 512), Image.ANTIALIAS)
            label = label.resize((512, 512), Image.ANTIALIAS)

            corrupt_image1.save(os.path.join(path_to_save_t0, "{}.png".format(imgname[0])))
            corrupt_image2.save(os.path.join(path_to_save_t1, "{}.png".format(imgname1[0])))
            label.save(os.path.join(path_to_save_label, "{}.png".format(m1[0])))



        # f, axarr = plt.subplots(1, 3)
        # # axarr[0, 0].imshow(i1)
        # axarr[0].imshow(corrupt_image1)
        # axarr[1].imshow(corrupt_image2)
        # axarr[2].imshow(label)

        # plt.show()
