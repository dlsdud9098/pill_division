from multiprocessing import Process, freeze_support
from multiprocessing import Pool, Manager
from itertools import repeat
import pandas as pd
import urllib.request as req

data = pd.read_csv('D:\download\OpenData_PotOpenTabletIdntfcC20230209.csv')
# data = data.iloc[:10,:]

def image_download(row):
    try:
        req.urlretrieve(row['큰제품이미지'], 'D:/download/images/'+str(row['품목일련번호'])+'.png')
        print(row['품목일련번호'], '이미지 생성')
    except:
        pass

        

if __name__ == '__main__':
    list_b = []
    for idx, row in data[['품목일련번호', '큰제품이미지']].iterrows():
        list_b.append(row)

    freeze_support()

    pool = Pool(processes=4)
    pool.starmap(image_download, zip(list_b))

    pool.close()