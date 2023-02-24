# d

# # 이미지 파일 리스트 생성
# pill_images = glob('./pill_images/*')
# # 리스트를 배치 크기만큼 나누기
# batch_size = 1000
# batch_list = [pill_images[i:i+batch_size] for i in range(0, len(pill_images), batch_size)]

# def load_image(pill):
#     """
#     이미지 파일을 읽어서 numpy 배열로 변환하는 함수
#     """
#     idx = os.path.basename(pill)[:-4]
#     img = Image.open(pill)
#     x = np.array(img)
#     return {'품목일련번호': idx, '이미지': x}

# def load_images(batch):
#     """
#     배치 내 모든 이미지 파일을 읽어서 numpy 배열로 변환하는 함수
#     """
#     with Pool(processes=4) as pool:
#         results = pool.map(load_image, batch)
#     return results

# if __name__ == '__main__':
#     freeze_support()

#     # 배치 크기마다 이미지 파일 읽어오기
#     pill_image_list = []
#     for batch in batch_list:
#         results = load_images(batch)
#         pill_image_list += results

#     # 결과를 pandas 데이터프레임으로 변환하여 CSV 파일로 저장
#     if len(pill_image_list) > 0:
#         pill_image_csv = pd.DataFrame(pill_image_list)
#         pill_image_csv.to_csv('pill_images.csv', index=False)
#     else:
#         print('이미지 파일을 처리하지 못했습니다.')
    

from multiprocessing import Process, freeze_support
from multiprocessing import Pool, Manager
from itertools import repeat
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import os
import pickle

# 이미지 파일 리스트 생성
pill_images = glob('./pill_images/*')
# 리스트를 배치 크기만큼 나누기
batch_size = 1000
batch_list = [pill_images[i:i+batch_size] for i in range(0, len(pill_images), batch_size)]

def load_image(pill):
    """
    이미지 파일을 읽어서 numpy 배열로 변환하는 함수
    """
    idx = os.path.basename(pill)[:-4]
    img = Image.open(pill)
    x = np.array(img)
    return {'품목일련번호': idx, '이미지': x}

def load_images(batch):
    """
    배치 내 모든 이미지 파일을 읽어서 numpy 배열로 변환하는 함수
    """
    with Pool(processes=4) as pool:
        results = pool.map(load_image, batch)

    # 결과 직렬화
    serialized_results = []
    for result in results:
        serialized_result = {}
        for k, v in result.items():
            if k == '이미지':
                serialized_result[k] = pickle.dumps(v)
            else:
                serialized_result[k] = v
        serialized_results.append(serialized_result)

    # 직렬화된 결과 복원
    restored_results = []
    for result in serialized_results:
        restored_result = {}
        for k, v in result.items():
            if k == '이미지':
                restored_result[k] = pickle.loads(v)
            else:
                restored_result[k] = v
        restored_results.append(restored_result)
    
    return restored_results

if __name__ == '__main__':
    global batch
    freeze_support()

    # 배치 크기마다 이미지 파일 읽어오기
    pill_image_list = []
    for batch in batch_list:
        results = load_images(batch)
        pill_image_list += results

    # 결과를 pandas 데이터프레임으로 변환하여 CSV 파일로 저장
    pill_image_csv = pd.DataFrame(pill_image_list)
    pill_image_csv.to_csv('pill_images.csv', index=False)