from .helper import *
import random
from sklearn.linear_model import LogisticRegression

from PIL import Image

from .celeb_utils import filter_images_by_attribute
import matplotlib.pyplot as plt

from torchvision import transforms
import numpy as np

data_dir = 'celeba_data/'
download_extract('celeba', data_dir)

random.seed(0)

width = 64
height = 64

# Images with only glasses
glasses_ids = filter_images_by_attribute(
    data_dir=data_dir,
    attr_dir = data_dir,
    attr1='Eyeglasses',
    present1=True,
    attr2='Wearing_Hat',
    present2=False
)

hat_ids = filter_images_by_attribute(
    data_dir=data_dir,
    attr_dir = data_dir,
    attr1='Eyeglasses',
    present1=False,
    attr2='Wearing_Hat',
    present2=True
)

bg_ids = filter_images_by_attribute(
    data_dir=data_dir,
    attr_dir = data_dir,
    attr1='Eyeglasses',
    present1=False,
    attr2='Wearing_Hat',
    present2=False
)


tg_ids = hat_ids + glasses_ids
random.shuffle(tg_ids)

show_n_images = 4
celeb_images = get_batch(bg_ids[:show_n_images], width, height, 'RGB')


img = celeb_images[0]/255

imgs = images_square_grid(celeb_images, 'RGB')


print('bg_ids len : ',len(bg_ids))

print('tg_ids len : ', len(tg_ids))

num_hat_samples = 5000
num_glasses_samples = 5000
num_background_samples = 10000
total_ids = bg_ids[:num_background_samples] + glasses_ids[:num_glasses_samples] + hat_ids[:num_hat_samples]
total_labels = np.concatenate([
    np.zeros(num_background_samples),
    np.ones(num_glasses_samples),
    np.ones(num_hat_samples) * 2,
])

np.save("celeba_ids.npy", total_ids)
np.save("celeba_labels.npy", total_labels)


num_hat_samples_test = 32
num_glasses_samples_test = 32
num_background_samples_test = 64
total_ids_test = bg_ids[num_background_samples:num_background_samples+num_background_samples_test] +\
        glasses_ids[num_glasses_samples:num_glasses_samples+num_glasses_samples_test] +\
        hat_ids[num_hat_samples:num_hat_samples+num_hat_samples_test]

total_labels_test = np.concatenate([
    np.zeros(num_background_samples_test),
    np.ones(num_glasses_samples_test),
    np.ones(num_hat_samples_test) * 2,
])

np.save("celeba_ids_test.npy", total_ids_test)
np.save("celeba_labels_test.npy", total_labels_test)


num_hat_samples_eval = 5000
num_glasses_samples_eval = 5000
num_background_samples_eval = 1000
total_ids_eval = bg_ids[num_background_samples+num_background_samples_test:num_background_samples+num_background_samples_test+num_background_samples_eval] +\
        glasses_ids[num_glasses_samples+num_glasses_samples_test:num_glasses_samples+num_glasses_samples_test+num_glasses_samples_eval] +\
        hat_ids[num_hat_samples+num_hat_samples_test:num_hat_samples+num_hat_samples_test+num_hat_samples_eval]

total_labels_eval = np.concatenate([
    np.zeros(num_background_samples_eval),
    np.ones(num_glasses_samples_eval),
    np.ones(num_hat_samples_eval) * 2,
])

np.save("celeba_ids_eval.npy", total_ids_eval)
np.save("celeba_labels_eval.npy", total_labels_eval)


