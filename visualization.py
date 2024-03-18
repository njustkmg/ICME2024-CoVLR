import json
import os

import matplotlib.pyplot as plt

img2img = json.load(open('img2img_top11.json'))
text2text = json.load(open('text2text_top11.json'))
# os.makedirs('examples')

# for i in range(len(text2text)):
#     example = text2text[list(text2text.keys())[i + 100]]
#     print(example)

for i in range(len(img2img)):

    example = img2img[list(img2img.keys())[i+100]]
    print(example[0])

    img_dir = 'data/flickr30k'

    img_0 = plt.imread(f"{img_dir}/{example[0]}")
    img_1 = plt.imread(f"{img_dir}/{example[1]}")
    img_2 = plt.imread(f"{img_dir}/{example[2]}")
    img_3 = plt.imread(f"{img_dir}/{example[3]}")
    img_4 = plt.imread(f"{img_dir}/{example[4]}")
    img_5 = plt.imread(f"{img_dir}/{example[5]}")
    img_6 = plt.imread(f"{img_dir}/{example[6]}")
    img_7 = plt.imread(f"{img_dir}/{example[7]}")
    img_8 = plt.imread(f"{img_dir}/{example[8]}")
    # img_9 = plt.imread(f"{img_dir}/{example[9]}")
    # img_10 = plt.imread(f"{img_dir}/{example[10]}")

    plt.subplot(261, xlabel=f"{example[0]}")
    plt.imshow(img_0)
    plt.subplot(262, xlabel='top_1')
    plt.imshow(img_1)
    plt.subplot(263, xlabel='top_2')
    plt.imshow(img_2)
    plt.subplot(264, xlabel='top_3')
    plt.imshow(img_3)
    plt.subplot(265, xlabel='top_4')
    plt.imshow(img_4)
    plt.subplot(266, xlabel='top_5')
    plt.imshow(img_5)
    plt.subplot(267, xlabel='top_6')
    plt.imshow(img_6)
    plt.subplot(268, xlabel='top_7')
    plt.imshow(img_7)
    plt.subplot(269, xlabel='top_8')
    plt.imshow(img_8)
    # plt.subplot(2610, xlabel='(1)', title="K_VRH")
    # plt.imshow(img_0)
    # plt.subplot(2611, xlabel='(1)', title="K_VRH")
    # plt.imshow(img_0)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.savefig(f'./examples/exa_{i}.jpg')