from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np

image_ids = ['29', '32', '37', '44', '47_2', '70']

for image_id in image_ids:
    print(f'Processing #{image_id}')
    fig, axes = plt.subplots(5, 5, figsize=(15, 13))
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle(f'{image_id}.jpg (E=0.4)', fontsize=16)

    # Original image and output
    img_path = f'data/test_data/DICM/{image_id}.jpg'
    image = mpimg.imread(img_path)
    axes[0][0].imshow(image)
    axes[0][0].set_title(f'Original')

    img_path = f'data/result/DICM/{image_id}.jpg'
    image = mpimg.imread(img_path)
    axes[1][0].imshow(image)
    axes[1][0].set_title(f'Zero-DCE')

    for i, w_exp in enumerate(range(10, 35, 5)):
        for j, w_col in enumerate(range(5, 25, 5)):
            img_path = f'data/result/test_weights/exp{w_exp}_col{w_col}_04_{image_id}.jpg'
            image = mpimg.imread(img_path)
            axes[i][j+1].imshow(image)
            axes[i][j+1].set_title(f'w_exp: {w_exp}, w_col: {w_col}')
    
    plt.tight_layout()
    plt.savefig(f'comparison/comp_04_{image_id}.jpg')