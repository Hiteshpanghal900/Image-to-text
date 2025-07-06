import os
import cv2
import csv
import tarfile
import pandas as pd
from urllib.request import urlretrieve
from scipy.io import loadmat

imageSize = (64,256)
datasetURL = "https://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz"

# Directory paths for extraction
downloadPath = "temp_download/IIIT5K.tar.gz"
extractDir = "temp_download/IIIT5K"

# File paths for processed data
outputDir = "../data/english"
ProcessedImageDir = f"{outputDir}/images"
labelCsvPath = f"{outputDir}/labels.csv"



def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
    bar = ('#' * int(percent / 2)).ljust(50)
    print(f"\rDownloading: [{bar}] {percent}% ({downloaded // 1024} KB)", end='')




# Step 1: Download and extract dataset
def download_and_extract():
    print("--------------IIIT-5K Dataset----------------")
    os.makedirs("temp_download", exist_ok=True)

    if os.path.exists(downloadPath):
        print("Dataset already downloaded")
    else:
        print("Downloading ...")
        urlretrieve(datasetURL, downloadPath, reporthook=download_progress_hook)


    if os.path.exists(extractDir):
        print("Dataset already extracted")
    else:
        print("Extracting ...")
        with tarfile.open(downloadPath, "r:gz") as tar:
            tar.extractall("temp_download")

# Step 2: Preprocessing
def preprocess():
    print("Preprocessing...")
    os.makedirs(ProcessedImageDir, exist_ok=True)

    matPath = os.path.join(extractDir,"trainCharBound.mat")

    mat = loadmat(matPath)
    samples = mat["trainCharBound"][0]

    entries = []
    imgRoot = os.path.join(extractDir)

    for sample in samples:
        imgName = str(sample['ImgName'][0])
        label = str(sample['chars'][0]).strip()
        imgFile = os.path.basename(imgName)

        srcImgPath = os.path.join(imgRoot, imgName)             #Path where image is present
        outImgPath = os.path.join(ProcessedImageDir, imgFile)       #Path where image is to be saved

        if not os.path.exists(srcImgPath):
            continue

        img = cv2.imread(srcImgPath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        imgResized = cv2.resize(img, (imageSize[1], imageSize[0]))
        cv2.imwrite(outImgPath, imgResized)

        entries.append((imgFile, label))

    df = pd.DataFrame(entries, columns=["filename", "text"])
    df.to_csv(labelCsvPath, index=False, quoting=csv.QUOTE_ALL)

    print(f"Saved {len(df)} images")





if __name__ == "__main__":
    download_and_extract()
    preprocess()