"""
Takes pretrained model with the best model's weights and makes predictions on custom data.
Returns DataFrame like:
df = pd.DataFrame({'Image_filename': ...,
                    'y_pred': ...,
                    'predict_label': ...,
                    'probability': ...})
where 'Image_filename' - image path
      'y_pred' - Predict label int (from 0 to 100)
      'predict_label' - Predict label (apple_pie, baby_back_ribs, ...)
      'probability' - Probability of predict label
"""
import preprocess_data, engine
from tensorflow.keras import mixed_precision
import argparse
import pandas as pd
import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
CHECK_PATH = '/content/checkpoints/efficientnetb0_fine_tuning_model_checkpoint_weights/checkpoint'

parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", help='directory of test data')
parser.add_argument("--custom_data_dir", help='directory of custom data')
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help='batch size, default=32')
parser.add_argument("--img_size", type=int, default=IMG_SIZE, help='image size, default=224')
parser.add_argument("--check_path", type=str, default=CHECK_PATH, help='directory to save checkpoint data')

args = parser.parse_args()

custom_data = tf.data.Dataset.list_files(args.custom_data_dir + '/*', shuffle=False)
file_names = list(custom_data.as_numpy_iterator())
class_names = preprocess_data.get_class_names(args.test_dir)

custom_ds = custom_data.map(preprocess_data.decode_img_from_path, num_parallel_calls=tf.data.AUTOTUNE)
custom_ds = custom_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

# Load the model
print(f"Loading model's best weights from {args.check_path}")
mixed_precision.set_global_policy('mixed_float16')

base_model = tf.keras.applications.EfficientNetB0(include_top=False)

# Create a model 
model = engine.create_model(base_model=base_model, 
                            img_size=args.img_size, 
                            class_names=class_names)


# Load the best weights
model.load_weights(args.check_path)

predictions = model.predict(custom_ds)
y_pred = predictions.argmax(axis=1)
probability = predictions.max(axis=1)

print("Create dataframe with predictions on custom data...")
custom_df = pd.DataFrame({'Image_filename':file_names,
                            'y_pred':y_pred,
                            'predict_label':[class_names[i] for i in y_pred],
                            'probability':probability})

custom_df.to_csv('food101_custom.csv', index=False)
