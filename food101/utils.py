"""
Create custom confusion matrix
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

def make_confusion_matrix(y_true, y_pred, labels, figsize=(90,90), text_size=10):
  """
  Create custom confusion matrix

  Args:
    y_true - True labels
    y_pred - Model's predicted labels 
    labels - Name of labels
    figsize - Figure size, default=(90,90), 
    text_size - Size of text, default=10
  """
  conf_mat=confusion_matrix(y_true, y_pred)

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
  fig.colorbar(cax)
  ax.set(title='Consusion matrix',
         xlabel='Predicted label',
         ylabel='True label',
         xticks=np.arange(conf_mat.shape[0]),
         yticks=np.arange(conf_mat.shape[1]),
         xticklabels=labels,
         yticklabels=labels)
  
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom

  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  ax.yaxis.label.set_size(text_size)
  ax.xaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  threshold = (conf_mat.max() + conf_mat.min())/2

  for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
    plt.text(j, i, f'{conf_mat[i,j]}',
             color = 'white' if conf_mat[i,j] > threshold else 'black',
             size=text_size)

def view_image_with_predict(df:pd.DataFrame,
                            index:int):
  """
  Visualize 6 images and treir predicted labels

  Args:
    df - DataFrame with model's prediction (from get_results.py or predict_on_custom_data.py)
    index - index of first image to visualize
  """
  for i, row in enumerate(df[index:index+6].itertuples()):
    plt.subplot(2, 3, i+1)
    img = preprocess_data.decode_img_from_path(row[1])
    plt.imshow(img/255.)
    plt.title(f'Label: {row[3]}\n Prob={row[4]:.2f}')
    plt.axis(False) 
