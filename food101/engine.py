"""
Create and compile the model
"""
from tensorflow.keras import layers
import tensorflow as tf

def create_model(base_model, img_size, class_names):
  """
  Create and compile model 

  Args:
    base_model - pretrained model such Resnet, EficientNet and other
    img_size - size of input image
    class_names - list of class names

  Returns:
    model

  Usage example:
    model = create_model(base_model=resnet50_model, 
                        img_size=(224, 224),
                        class_names=class_names)
  """
  inputs = layers.Input(shape=img_size + (3,))
  x = base_model(inputs)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(len(class_names))(x)
  outputs = layers.Activation('softmax', dtype=tf.float32)(x)
  model = tf.keras.Model(inputs, outputs)

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])
  return model
