import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import glob
import zipfile

################################

def formatBeeData(data_path):
    with zipfile.ZipFile(data_path+'/DataSets/Bees/archive.zip', 'r') as zip_ref:
        zip_ref.extractall(data_path+'/DataSets/Bees/raw_data')

    os.rename(data_path+'/DataSets/Bees/raw_data/bee_data.csv',data_path+'/DataSets/Bees/raw_data/data_tab.csv' )

    df = pd.read_csv(data_path+'/DataSets/Bees/raw_data/data_tab.csv')
    df["ID"] = range(0,len(df))
    df = df.sample(frac=1).reset_index(drop=True)

    file_id = df["file"].tolist()

    os.mkdir(data_path+'/DataSets/Bees/raw_data/images')
    for i in range(len(file_id)):
        img = Image.open(data_path+'/DataSets/Bees/raw_data/bee_imgs/bee_imgs/'+file_id[i])
        img.save(data_path+'/DataSets/Bees/raw_data/images/'+str(i)+".png")

    df.to_csv(data_path+'/DataSets/Bees/raw_data/data_tab.csv')
    shutil.rmtree(data_path+'/DataSets/Bees/raw_data/bee_imgs')

################################

def formatMonkeyData(data_path):
    with zipfile.ZipFile(data_path+'/DataSets/Monkey/archive.zip', 'r') as zip_ref:
        zip_ref.extractall(data_path+'/DataSets/Monkey/raw_data')

    df = pd.read_csv(data_path+'/DataSets/Monkey/raw_data/monkey_labels.txt')

    # Copy training images
    os.mkdir(data_path+'/DataSets/Monkey/raw_data/images')

    train_dir = ['n0', 'n7', 'n9', 'n8', 'n6', 'n1', 'n4', 'n3', 'n2', 'n5']
    train_dir.sort()

    all_labels = []
    i, j = 0, 0
    for path in train_dir:
        file_names = glob.glob(os.path.join(data_path+'/DataSets/Monkey/raw_data'+'/training/training',path)+"/*.jpg")
        names = [name.split("/")[-1] for name in file_names]
        for pic_path in file_names:
            img = Image.open(pic_path)
            img.save(data_path+'/DataSets/Monkey/raw_data/'+"images/"+str(i)+".jpg")
            i += 1
        labels = [j]*len(file_names)
        all_labels.append(labels)
        j += 1
    labels = [c_sub for c in all_labels for c_sub in c ]

    train_df = {"ID": range(0,len(labels)),"class_id":labels}
    train_df = pd.DataFrame(train_df)

    # Copy training images
    valid_dir = ['n0', 'n7', 'n9', 'n8', 'n6', 'n1', 'n4', 'n3', 'n2', 'n5']
    valid_dir.sort()

    all_labels = []
    j =  0
    val_ids = []
    for path in train_dir:
        file_names = glob.glob(os.path.join(data_path+'/DataSets/Monkey/raw_data'+"/validation/validation",path)+"/*.jpg")
        names = [name.split("/")[-1] for name in file_names]
        for pic_path in file_names:
            img = Image.open(pic_path)
            img.save(data_path+'/DataSets/Monkey/raw_data/'+"images/"+str(i)+".jpg")
            val_ids.append(i)
            i += 1
        labels = [j]*len(file_names)
        all_labels.append(labels)
        j += 1
    labels = [c_sub for c in all_labels for c_sub in c ]

    valid_df = {"ID": val_ids, "class_id":labels}
    valid_df = pd.DataFrame(valid_df)
    valid_df.head()

    data_tab = pd.concat([train_df,valid_df],ignore_index=True)
    data_tab = data_tab.sample(frac=1).reset_index(drop=True)
    data_tab.to_csv(data_path+'/DataSets/Monkey/raw_data/'+"data_tab.csv")

    # Resize images
    img_names = glob.glob(data_path+'/DataSets/Monkey/raw_data/'+"images/*.jpg")
    for name in img_names:
        img = Image.open(name).convert('RGB')
        img = img.resize((120,120))
        img.save(name)

    shutil.rmtree(data_path+'/DataSets/Monkey/raw_data/training')
    shutil.rmtree(data_path+'/DataSets/Monkey/raw_data/validation')
    os.remove(data_path+'/DataSets/Monkey/raw_data/monkey_labels.txt')

################################

def formatMNISTData(data_path):
    with zipfile.ZipFile(data_path+'/DataSets/MNIST/archive.zip', 'r') as zip_ref:
        zip_ref.extractall(data_path+'/DataSets/MNIST/raw_data')

    df_train = pd.read_csv(data_path+'/DataSets/MNIST/raw_data/mnist_train.csv')
    df_test = pd.read_csv(data_path+'/DataSets/MNIST/raw_data/mnist_test.csv')
    data_tab = pd.concat([df_train,df_test],ignore_index=True)

    data_tab.insert(0, 'ID', range(0,len(data_tab)))
    data_tab.to_csv(data_path+'/DataSets/MNIST/raw_data/'+"mnist.csv",index=False)

    data_tab = data_tab.iloc[:,:2]
    data_tab.to_csv(data_path+'/DataSets/MNIST/raw_data/'+"data_tab.csv",index=False)

    os.remove(data_path+'/DataSets/MNIST/raw_data/mnist_train.csv')
    os.remove(data_path+'/DataSets/MNIST/raw_data/mnist_test.csv')

################################

def main():
    data_path = 'Data'
    formatBeeData(data_path)
    formatMonkeyData(data_path)
    formatMNISTData(data_path)

################################

if __name__ == "__main__":
    main()
