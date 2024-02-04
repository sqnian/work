
## create new file
```
mkdir flower_data
cd flower_data
wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar -zxvf flower_photos.tgz
cd ..
```

```
├── flower_data   
       ├── flower_photos（解压的数据集文件夹，3670个样本）  
       ├── train（生成的训练集，3306个样本）  
       └── val（生成的验证集，364个样本） 
```

## split data

```
python3 split_data.py
```


