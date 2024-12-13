from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import glob

img_paths = glob.glob('data/test_data/Over/*.jpg')

for img_path in img_paths:
    image_name = img_path.split('/')[-1]
    print(f'Processing #{image_name}')
    fig, axes = plt.subplots(5, 5, figsize=(15, 13))
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle(f'{image_name} (E=0.3)', fontsize=16)

    # Original image and output
    image = mpimg.imread(img_path)
    axes[0][0].imshow(image)
    axes[0][0].set_title(f'Original')

    for i, w_exp in enumerate(range(10, 35, 5)):
        for j, w_col in enumerate(range(5, 25, 5)):
            img_path = f'data/result/test_weights3/exp{w_exp}_col{w_col}/{image_name}'
            image = mpimg.imread(img_path)
            axes[i][j+1].imshow(image)
            axes[i][j+1].set_title(f'w_exp: {w_exp}, w_col: {w_col}')
    
    plt.tight_layout()
    plt.savefig(f'comparison/comp_fuse_{image_name}')