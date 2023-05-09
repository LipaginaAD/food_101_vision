"""
Prepare data for training.
"""
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf
import os

def unzip_train_test_data(zip_file_dir:str,
                          data_path:str,
                          train_or_test:str):
  """
  Args:
    zip_file_dir: str - Directory of zip file (in format: 'your_dir/filename.zip')
    data_path: str - Directory to unzip files
    train_or_test: str - Name of dataset: train or test

  Return:
    A directory of unzipped data

  Usage example:
    directory = unzip_train_test_data(zip_file_dir='your_directory/zip_file_name.zip',
                                      data_path='/content/data',
                                      train_or_test='train')
  """
  data_path = Path(data_path)
  data_dir = data_path / train_or_test
  
  # Create directory
  if data_dir.is_dir():
    print(f'{data_dir} directory already exist')
  else:
    print(f'Create {data_dir} directory')
    data_dir.mkdir(parents=True, exist_ok=True)

  with zipfile.ZipFile(zip_file_dir, 'r') as zip_ref:
    print("Unzipping data...") 
    zip_ref.extractall(data_dir)
  os.remove(zip_file_dir)

  return str(data_dir)

def get_class_names(data_dir):
  """
  Get class names from directory with data
  """
  data_dir = Path(data_dir)
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
  return class_names

def get_label_from_path(file_path, class_names):
  """
  Get labels from image path 
  """
  parts = tf.strings.split(file_path, sep='/')
  label = parts[-2] == class_names
  return tf.argmax(label)

def decode_img_from_path(file_path):
  """
  Read, decode and resize image from image path
  """
  img = tf.io.read_file(file_path)
  img = tf.io.decode_jpeg(img, channels=3)
  return tf.image.resize(img, size=(224, 224))

def process_path(file_path, class_names):
  """
  Returns tuple (image, label) with decode image and label from image path
  """
  img = decode_img_from_path(file_path)
  label = get_label_from_path(file_path, class_names)
  return img, label

def create_prefetch_dataset(data_dir:str,
                            shuffle:bool,
                            batch_size:int):
  """
  Create prefetch datasets

  Args:
    data_dir - directory of data
    shuffle - True for suffle data, False for unshuffle data
    batch_size - number of samples per batch

  Returns:
    Prefetch dataset

  Usage example:
    train_dataset = create_prefetch_dataset(data_dir='your_path/train_data',
                                            shuffle=True,
                                            batch_size=32) 
  """
  class_names =  get_class_names(data_dir)                        
  if shuffle:
    print('Create train prefetch dataset...')
    data = tf.data.Dataset.list_files(data_dir  + '/*/*')
    dataset = data.map(lambda x: process_path(x, class_names=class_names), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  else:
    print('Create test prefetch dataset...')
    data = tf.data.Dataset.list_files(data_dir  + '/*/*', shuffle=False)
    dataset = data.map(lambda x: process_path(x, class_names=class_names), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return dataset

