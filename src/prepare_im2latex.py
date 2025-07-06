import os
import cv2
import csv
import pandas as pd
from urllib.request import urlretrieve
from zipfile import ZipFile

imageSize = (64,256)
datasetURL = "https://github.com/harvardnlp/im2markup/archive/refs/heads/master.zip"

# Directory paths for extraction
extractDir = "temp_download/im2markup-master"
imgDir = f"{extractDir}/data/sample/images"
labelPath = f"{extractDir}/data/sample/formulas.lst"

# File paths for processed data
outputDir = "../data/math"
processedImageDir = f"{outputDir}/images"
processedLabelCSV = f"{outputDir}/labels.csv"

# Step 1: Download and extract dataset
def download_and_extract():
    if os.path.exists(imgDir) and os.path.exists(labelPath):
        print("Dataset already downloaded")
        return 
    

    print("Downloading dataset..")
    os.makedirs("temp_download", exist_ok=True)
    zipPath = "temp_download/im2latex.zip"
    urlretrieve(datasetURL, zipPath)

    print("Retrieving Data..")
    with ZipFile(zipPath, 'r') as file:
        file.extractall("temp_download")

# Step 2: Preprocess
def preprocess():
    if os.path.exists(processedImageDir) and os.path.exists(processedLabelCSV):
        print("✅ Dataset already preprocessed.")
        return

    print("🔧 Processing images and LaTeX...")

    os.makedirs(processedImageDir, exist_ok=True)

    # Read formulas
    with open(labelPath, 'r', encoding='utf-8') as f:
        formulas = [line.strip() for line in f.readlines()]

    # Read train.lst (or test.lst/val.lst)
    mapping_path = "temp_download/im2markup-master/data/sample/train.lst"
    entries = []

    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            formula_idx = int(parts[0])
            img_name = parts[1] + ".png"

            if formula_idx >= len(formulas):
                continue

            formula = formulas[formula_idx]
            src_img_path = os.path.join(imgDir, img_name)

            if not os.path.exists(src_img_path):
                continue

            img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (imageSize[1], imageSize[0]))
            cv2.imwrite(os.path.join(processedImageDir, img_name), img_resized)

            entries.append((img_name, formula))

    df = pd.DataFrame(entries, columns=["filename", "latex"])
    df.to_csv(processedLabelCSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Saved {len(df)} image-label pairs.")



if __name__ == "__main__":
    download_and_extract()
    preprocess()