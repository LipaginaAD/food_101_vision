# food_101_vision

This computer vision classification model tekes food photos and predict name of dishes.


## Data
Data taken from Kaggle: https://www.kaggle.com/datasets/kmader/food41
Dataset contains 101 categories of different types of food.

## Usage

### To prepare data from Kaggle to working with it:

```
from food101 import prepare_data
prepare_data.unzip_data(target_directory='your_path',
                        dataset_name='food_101',
                        zip_file_name='archive.zip')

prepare_data.split_train_test_zip_it('/your_path/food_101')
```


### To train model:

```
!python food101/train.py --train_zip_file_dir='/your_path/train.zip' --test_zip_file_dir='/your_path/test.zip'
```

You can also change the following settings:

--data_path - directory to unziped files

--epochs - number of epochs, default=100

--batch_size - number of samples per batch, default=32

--check_path - directory to save checkpoint data

--img_size - image size,  default=(224, 224) 


### To test model on test data:


```
!python food101/get_results.py --test_dir='/your_path/food_data/test' --check_path='/your_check_path/checkpoint'
```

You can also change the following settings:

--batch_size - number of samples per batch, default=32

--img_size - image size,  default=(224, 224) 

### To use model on custom food images: 

```
!python food101/predict_on_custom_data.py --test_dir='/your_path/food_data/test' --custom_data_dir='/your_path/custom_data' --check_path='/your_check_path/checkpoint'
```

You can also change the following settings:

--batch_size - number of samples per batch, default=32

--img_size - image size,  default=(224, 224) 
