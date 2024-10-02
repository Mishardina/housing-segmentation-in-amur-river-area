import os
import glob
import pandas as pd
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

CPU_COUNT = os.cpu_count()

CURR_DIR = os.getcwd()
#DATA_DIR = os.path.abspath('../data')
DATA_DIR = os.path.join(CURR_DIR, 'data')
INRIA_DIR = os.path.join(DATA_DIR, 'AerialImageDataset')

def augment_image_and_mask(number_of_tiles, image_path, mask_path, image_name, tiled_images_path, tiled_masks_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    h_image, w_image = img.shape[:2]
    h_mask, w_mask = mask.shape[:2]
    assert h_image == h_mask and w_image == w_mask

    h, w = h_image, w_image
    number_of_tiles = 10
    h_tile, w_tile = h // number_of_tiles, w // number_of_tiles

    tiles = []
    for y in range(number_of_tiles):
        y_start = y * h_tile
        y_end = y_start + h_tile

        y_start = int(y_start)
        y_end = int(y_end)

        for x in range(number_of_tiles):
            x_start = x * w_tile
            x_end = x_start + w_tile

            x_start = int(x_start)
            x_end = int(x_end)

            img_tile = img[y_start:y_end, x_start:x_end]
            mask_tile = mask[y_start:y_end, x_start:x_end]
            
            tiles.append((img_tile, mask_tile))

    for i in range(len(tiles)):
        cv2.imwrite(os.path.join(tiled_images_path, f'{image_name}_image_{i}.jpg'), img=tiles[i][0])
        cv2.imwrite(os.path.join(tiled_masks_path, f'{image_name}_mask_{i}.png'), img=tiles[i][1])

INRIA_EXTRACTED_PATH = os.path.join(DATA_DIR, 'PreprocessedAerialImageDataset')

for subset in ['train']:
    INRIA_IMAGES_PATH = os.path.join(INRIA_DIR, subset, 'images')
    INRIA_LABELS_PATH = os.path.join(INRIA_DIR, subset, 'gt')

    images_paths = glob.glob(os.path.join(INRIA_IMAGES_PATH, '*.tif'))
    print(INRIA_IMAGES_PATH)
    images_paths = [x.replace('\\', '/') for x in images_paths]
    images_names = [x.split('/')[-1].split('.')[0] for x in images_paths]

    masks_paths = glob.glob(os.path.join(INRIA_LABELS_PATH, '*.tif'))
    masks_paths = [x.replace('\\', '/') for x in masks_paths]
    masks_names = [x.split('/')[-1].split('.')[0] for x in masks_paths]

    images_dict = {'images_paths': images_paths, 'images_names': images_names}
    masks_dict = {'masks_paths': masks_paths, 'masks_names': masks_names}

    df_images = pd.DataFrame(data=images_dict)
    df_masks = pd.DataFrame(data=masks_dict)
    df_inria = pd.merge(df_images, df_masks, how='inner', left_on='images_names', right_on='masks_names')

    INRIA_TILED_IMAGES_PATH = os.path.join(INRIA_EXTRACTED_PATH, subset, 'images')
    INRIA_TILED_MASKS_PATH = os.path.join(INRIA_EXTRACTED_PATH, subset, 'gt')

    if not os.path.exists(INRIA_TILED_IMAGES_PATH):
        os.makedirs(INRIA_TILED_IMAGES_PATH)

    if not os.path.exists(INRIA_TILED_MASKS_PATH):
        os.makedirs(INRIA_TILED_MASKS_PATH)

    _ = Parallel(n_jobs=CPU_COUNT)(delayed(augment_image_and_mask)(10, df_inria.iloc[i]['images_paths'], df_inria.iloc[i]['masks_paths'], df_inria.iloc[i]['images_names'], INRIA_TILED_IMAGES_PATH, INRIA_TILED_MASKS_PATH) for i in tqdm(range(len(df_inria))))

    images_paths = glob.glob(os.path.join(INRIA_TILED_IMAGES_PATH, '*.jpg'))
    images_paths = [x.replace('\\', '/').replace('\\', '/') for x in images_paths]
    images_names = ['/'.join([x.split('/')[-2], x.split('/')[-1]]) for x in images_paths]

    masks_paths = glob.glob(os.path.join(INRIA_TILED_MASKS_PATH, '*.png'))
    masks_paths = [x.replace('\\', '/') for x in masks_paths]
    masks_names = ['/'.join([x.split('/')[-2], x.split('/')[-1]]) for x in masks_paths]

    images_dict = {'image': images_names}
    masks_dict = {'mask': masks_names}

    df_images = pd.DataFrame(data=images_dict)
    df_masks = pd.DataFrame(data=masks_dict)
    df_inria = pd.concat([df_images, df_masks], axis=1)

    df_inria.to_csv(os.path.join(INRIA_EXTRACTED_PATH, subset, 'labels.csv'), index=False)
    print(f"Subset {subset} preprocessed successfully")