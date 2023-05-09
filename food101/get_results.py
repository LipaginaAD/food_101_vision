"""
Takes pretrained model with the best model's weights and makes predictions on test data.
Returns DataFrame like:
df = pd.DataFrame({'Image_filename': ...,
                    'y_pred': ...,
                    'predict_label': ...,
                    'probability': ...,
                    'y_true': ...,
                    'true_label': ...})
where 'Image_filename' - image path
      'y_true' - True label int (from 0 to 100)
      'y_pred' - Predict label int (from 0 to 100)
      'true_label' - True label (apple_pie, baby_back_ribs, ...)
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
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help='batch size, default=32')
parser.add_argument("--img_size", type=int, default=IMG_SIZE, help='image size, default=224')
parser.add_argument("--check_path", type=str, default=CHECK_PATH, help='directory to save checkpoint data')

args = parser.parse_args()

test_data = tf.data.Dataset.list_files(args.test_dir + '/*/*', shuffle=False)
file_names = list(test_data.as_numpy_iterator())
class_names = preprocess_data.get_class_names(args.test_dir)
test_ds = preprocess_data.create_prefetch_dataset(data_dir=args.test_dir,
                                                  shuffle=False,
                                                  batch_size=args.batch_size)

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

y_true = np.array(list(test_data.map(lambda x: preprocess_data.get_label_from_path(x, class_names=class_names))))
predictions = model.predict(test_ds)
y_pred = predictions.argmax(axis=1)
probability = predictions.max(axis=1)

print("Create dataframe with model's results...")
results_df = pd.DataFrame({'Image_filename':file_names,
                            'y_pred':y_pred,
                            'predict_label':[class_names[i] for i in y_pred],
                            'probability':probability,
                            'y_true':y_true,
                            'true_label':[class_names[i] for i in y_true],})

results_df.to_csv('food101_results.csv', index=False)
