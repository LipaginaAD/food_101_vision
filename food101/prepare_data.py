"""
Prepare data to work with it. 
Unzip data downloaded from Kaggle and create zip files with train and test data to easy working.  

Usage example:
  from food101 import prepare_data
  prepare_data.unzip_data(target_directory='/content',
                          dataset_name='food_101',
                          zip_file_name='archive.zip')

  prepare_data.split_train_test_zip_it('/content/food_101')

"""
import zipfile
from pathlib import Path
import json
import os
import shutil

def unzip_data(target_directory:str,
               dataset_name:str,
               zip_file_name:str):
  """
  Unzip data file downloaded from Kaggle

  Args:
    target_directory: str - directory where zip file is
    dataset_name: str - Name of dataset to copy files
    zip_file_name: str - Name of zip file (in format: 'filename.zip')
  
  Return:
    Unzipped data in directory 'target_directory / dataset_name'

  Usage example:
    unzip_data(target_directory='data',
                dataset_name='name_of_dataset',
                zip_file_name='zip_file_name.zip')

  """
  directory = Path(target_directory)
  data_path = directory / dataset_name
  
  # Create directory
  if data_path.is_dir():
    print(f'{data_path} directory already exist')
  else:
    print(f'Create {data_path} directory')
    data_path.mkdir(parents=True, exist_ok=True)
    
  with zipfile.ZipFile(directory / zip_file_name, 'r') as zip_ref:
    print("Unzipping data...") 
    zip_ref.extractall(data_path)
  os.remove(directory / zip_file_name)

def get_labels(path):
  with open(path) as f:
    return json.load(f)

def split_train_test_zip_it(data_path):
  """
  Split data into train and test folders and zip it.

  Example of file structure
  food_101 <- top level folder
  └───train <- training images
  │   └───apple_pie
  │   │   │   1008104.jpg
  │   │   │   1638227.jpg
  │   │   │   ...      
  │   └───steak
  │   |   │   1000205.jpg
  │   |   │   1647351.jpg
  │   |   │   ...
  |   |
  |   |___...
  |   |
  |   |___waffles
  │   |   │   1000534.jpg
  │   |   │   1085675.jpg
  │   |   │   ...
  │   
  └───test <- testing images
  │   └───apple_pie
  │   │   │   1001116.jpg
  │   │   │   1507019.jpg
  │   │   │   ...      
  │   └───steak
  │   |   │   100274.jpg
  │   |   │   1653815.jpg
  │   |   │   ...  
  |   |
  |   |___...
  |   |
  |   |___waffles
  │   |   │   1000534.jpg
  │   |   │   1085675.jpg
  │   |   │   ... 
  """
  # Get train and test labels
  train_labels = get_labels(data_path + '/meta/meta/train.json')
  test_labels = get_labels(data_path + '/meta/meta/test.json')
  # Get classes
  classes = [line.split('\n')[0] for line in open(data_path + '/meta/meta/classes.txt')]
  
  # Create folder with test data
  print('Create test data...')
  for i in classes:
    os.makedirs(data_path + '/test/' + i)
    for j in test_labels[i]:
      original_path = data_path + '/images/' + j +'.jpg'
      new_path = data_path + '/test/' + j +'.jpg'
      shutil.copy2(original_path, new_path)
 
  # Create folder with train data
  print('Create train data...')
  for i in classes:
    os.makedirs(data_path + '/train/' + i)
    for j in train_labels[i]:
      original_path = data_path + '/images/' + j +'.jpg'
      new_path = data_path + '/train/' + j +'.jpg'
      shutil.copy2(original_path, new_path)
  
  # Zip train and test folders
  print('Zip train data...')
  shutil.make_archive('train', 'zip', data_path + '/train')
  print('Zip test data...')
  shutil.make_archive('test', 'zip', data_path + '/test')
