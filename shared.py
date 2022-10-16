"""
file containing shared functions
"""

from time import sleep
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def countdown(c):
    """
    function performs a count down from c to 1
    """
    for i in range(c)[::-1]:
        sleep(1)
        print(i + 1)


def show_images(df, label, rows=1, no_of_image=20):
    """
    df - input dataframe
    label - 0 : left , 1 : forward , 2 : left
    no_of_image - no of images that we want to visualize
    """
    # extract rows which has a particular disease name
    img_df = df[df["target"] == label]

    # take sample
    img_sample_df = img_df.sample(no_of_image)

    # get the image-name and disease name
    images = img_sample_df['path'].values

    fig = plt.figure()
    for n, img_id in enumerate(images):
        a = fig.add_subplot(rows, np.ceil(no_of_image / float(rows)), n + 1)
        # read a image
        img = Image.open(img_id)
        # plot the current image
        plt.imshow(img)
        a.set_title(n)
    fig.set_size_inches(30, 30)
    plt.show()


def prob_to_vector(a):
    v = np.zeros_like(a)
    v[np.argmax(a)] = 1
    return v.tolist()
