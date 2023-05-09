"""
Create datasets and model where EfficientNetB0 is base_model.
Train model with mixed precision with freeze base_model's layer for 5 epochs, then train model with 
unfreeze layers.

The best model's weights saved in check_path.
"""
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import tensorflow as tf
import preprocess_data, engine
import argparse

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 100
CHECK_PATH = '/content/checkpoints/efficientnetb0_fine_tuning_model_checkpoint_weights/checkpoint'
SAVED_MODEL_DIR = '/content/food101/save_model'

parser = argparse.ArgumentParser()
parser.add_argument("--train_zip_file_dir", help='directory of zip file with train data')
parser.add_argument("--test_zip_file_dir", help='directory of zip file with test data')
parser.add_argument("--data_path", default='/content/food_data', help='directory to unziped files')
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help='batch size, default=32')
parser.add_argument("--epochs", type=int, default=EPOCHS, help='number of epochs, default=100')
parser.add_argument("--img_size", type=int, default=IMG_SIZE, help='image size, default=224')
parser.add_argument("--check_path", type=str, default=CHECK_PATH, help='directory to save checkpoint data')

args = parser.parse_args()

# Unzip data
train_dir = preprocess_data.unzip_train_test_data(zip_file_dir=args.train_zip_file_dir,
                                                  data_path=args.data_path,
                                                  train_or_test = 'train')
test_dir = preprocess_data.unzip_train_test_data(zip_file_dir=args.test_zip_file_dir,
                                                  data_path=args.data_path,
                                                  train_or_test = 'test')
# Setup class names
class_names = preprocess_data.get_class_names(train_dir)

# Create datasets
train_ds = preprocess_data.create_prefetch_dataset(data_dir=train_dir,
                                                   shuffle=True,
                                                   batch_size=args.batch_size)
test_ds = preprocess_data.create_prefetch_dataset(data_dir=test_dir,
                                                  shuffle=False,
                                                  batch_size=args.batch_size)


# Set global policy
mixed_precision.set_global_policy('mixed_float16')

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# Create a model 
model = engine.create_model(base_model=base_model, 
                            img_size=args.img_size, 
                            class_names=class_names) 

# Fit the model
print('Train model with freeze layers...')
history = model.fit(train_ds,
                    epochs=5,
                    steps_per_epoch=len(train_ds),
                    validation_data=test_ds,
                    validation_steps=int(0.2*len(test_ds)))

# Create ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.check_path,
                                                          save_weights_only=True,
                                                          save_best_only=True,
                                                          save_freq='epoch',
                                                          verbose=1)

# Create Early Stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=3)

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2,
                                                 patience=2,
                                                 verbose=1,
                                                 min_lr=1e-7)


# Unfreeze all layers
base_model.trainable = True

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])
# Fit the model
print('Train the fine tuning model...')
history_fine_tuning = model.fit(train_ds,
                                epochs=args.epochs,
                                steps_per_epoch=len(train_ds),
                                validation_data=test_ds,
                                validation_steps=int(0.2*len(test_ds)),
                                initial_epoch=history.epoch[-1],
                                callbacks=[checkpoint_callback, early_stopping, reduce_lr])
# Load the best weights
model.load_weights(args.check_path)

# Evaluate the model
print("Calculate model's accuracy...")
model_accuracy = model.evaluate(test_ds)
print(f'Model accuracy: {model_accuracy}')
