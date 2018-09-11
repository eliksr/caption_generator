import sys
import os
import pickle
from keras.preprocessing import image
from vgg16 import VGG16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input
import cv2
import dicom
import pandas as pd
from imgaug import augmenters as iaa
import random


sys.path.append('/Users/esror/PycharmProjects/caption_app/caption_app')
counter = 0
text_data_dir = '../Xray_text/no_overlap/front'
image_data_dir = '../Xray_Dataset/front'

# augmentation functions
flipper = iaa.Fliplr(1.0)
agn = iaa.AdditiveGaussianNoise(scale=0.1*255)
crop = iaa.Crop(px=(0, 10))
aug_lst = [flipper, agn, crop]


def preprocessing_image(arr):
    arr = arr.astype('float32')

    # 'RGB'->'BGR'
    arr = arr[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    arr[..., 0] -= mean[0]
    arr[..., 1] -= mean[1]
    arr[..., 2] -= mean[2]
    arr /= 255
    return arr


def aug_img(img):
    aug_indx = random.randint(0, 2)
    aug_func = aug_lst[aug_indx]
    aug_img = aug_func.augment_image(img)
    return aug_img


def load_dicom_img(path, target_size=224):
    ds = dicom.read_file(path)
    img = ds.pixel_array.astype(np.float32)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        maxI = np.amax(img)
        img = (2 ** int(np.ceil(np.log2(maxI - 1)))) - 1 - img

    img = cv2.resize(img, (target_size, target_size))
    img = np.repeat(img[..., None], 3, axis=2)
    return img


def load_encoding_model():
    model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
    return model


def get_encoding(model, image):
    global counter
    counter += 1
    pred = model.predict(image)
    pred = np.reshape(pred, pred.shape[1])
    print ("Encoding image: "+str(counter))
    print (pred.shape)
    return pred


def prepare_dataset():
    df_train_images = pd.read_csv(os.path.join(text_data_dir,'img_tr.csv'))

    df_test_images = pd.read_csv(os.path.join(text_data_dir,'img_te.csv'))

    f_train_dataset = open(os.path.join(text_data_dir,'xray_train_dataset.txt'),'w')
    f_train_dataset.write("image_id\tcaptions\n")
    f_train_images = open(os.path.join(text_data_dir,'xray_trainImages.txt'), 'w')

    f_test_dataset = open(os.path.join(text_data_dir,'xray_test_dataset.txt'),'w')
    f_test_dataset.write("image_id\tcaptions\n")
    f_test_images = open(os.path.join(text_data_dir,'xray_testImages.txt'), 'w')

    encoded_images = {}
    encoding_model = load_encoding_model()

    c_train = 0
    for i in range(len(df_train_images)):
        for j in range(df_train_images.iloc[i].do_aug):
            fname = df_train_images.iloc[i].file_name
            img = load_dicom_img(os.path.join(image_data_dir,fname))
            if j > 0:
                img = aug_img(img)
            img = preprocessing_image(img)
            img = np.expand_dims(img, axis=0)
            fname = fname + str(j)
            encoded_images[fname] = get_encoding(encoding_model, img)

            caption = "<start> "+str(df_train_images.iloc[i].terms)+" <end>"
            f_train_dataset.write(fname+"\t"+caption+"\n")
            f_train_images.write(fname+"\n")
            f_train_dataset.flush()
            f_train_images.flush()
            c_train += 1
    f_train_dataset.close()
    f_train_images.close()


    c_test = 0
    for i in range(len(df_test_images)):
        fname = df_test_images.iloc[i].file_name
        img = load_dicom_img(os.path.join(image_data_dir,fname))
        img = preprocessing_image(img)
        fname = fname
        img = np.expand_dims(img, axis=0)
        encoded_images[fname] = get_encoding(encoding_model, img)

        caption = "<start> "+str(df_test_images.iloc[i].terms)+" <end>"
        f_test_dataset.write(fname+"\t"+caption+"\n")
        f_test_images.write(fname + "\n")
        f_test_dataset.flush()
        f_test_images.flush()
        c_test += 1
    f_test_dataset.close()
    f_test_images.close()
    with open( os.path.join(text_data_dir, "encoded_images.p"), "wb" ) as pickle_f:
        pickle.dump( encoded_images, pickle_f )
    return [c_train, c_test]


if __name__ == '__main__':
    c_train, c_test = prepare_dataset()
    print ("Training samples = "+str(c_train))
    print ("Test samples = "+str(c_test))
