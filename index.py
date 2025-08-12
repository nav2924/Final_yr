import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall

IMAGE_SIZE = (220, 220)
LABEL_COLUMNS = [
    'Early Blight', 'Healthy', 'Late Blight', 'Leaf Miner', 'Leaf Mold',
    'Mosaic Virus', 'Septoria', 'Spider Mites', 'Yellow Leaf Curl Virus'
]

def z_score_normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    if std == 0: std = 1e-6
    return (image - mean) / std

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32)
    return z_score_normalize(image)

def load_images_from_dataframe(df):
    X, Y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = load_and_preprocess_image(row['filepath'])
            X.append(img)
            Y.append(row['labels'])
        except FileNotFoundError as e:
            print(e)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def prepare_dataframe(folder_path):
    df = pd.read_csv(os.path.join(folder_path, '_classes.csv'))
    df.columns = df.columns.str.strip()
    df['filename'] = df['filename'].str.strip()
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(folder_path, x).replace('\\', '/'))
    df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)
    df['labels'] = df[LABEL_COLUMNS].values.tolist()
    return df


print("Loading CSVs...")
df_train = prepare_dataframe("train")
df_valid = prepare_dataframe("valid")
df_test  = prepare_dataframe("test")


print("Loading images...")
X_train, Y_train = load_images_from_dataframe(df_train)
X_valid, Y_valid = load_images_from_dataframe(df_valid)
X_test, Y_test   = load_images_from_dataframe(df_test)


import keras_tuner as kt
from tensorflow.keras.optimizers import Adam

def model_builder(hp):
    model = Sequential()

    # Conv layer 1
    model.add(Conv2D(hp.Choice('conv1_filters', [16, 32]), (3, 3), activation='relu', input_shape=(220, 220, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Conv layer 2
    model.add(Conv2D(hp.Choice('conv2_filters', [32, 64]), (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Conv layer 3
    model.add(Conv2D(hp.Choice('conv3_filters', [64, 128]), (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Dense layers
    model.add(Dense(hp.Choice('dense1_units', [128, 256, 512]), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('dropout1', [0.25, 0.3, 0.4])))

    model.add(Dense(hp.Choice('dense2_units', [64, 128]), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('dropout2', [0.25, 0.3])))

    model.add(Dense(len(LABEL_COLUMNS), activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=5,
    directory='kt_trials',
    project_name='tomato_leaf_tuning'
)
tuner.search(X_train, Y_train,
             epochs=10,
             validation_data=(X_valid, Y_valid),
             batch_size=32)

best_hps = tuner.get_best_hyperparameters(1)[0]
print("Best Hyperparameters:")
for param in best_hps.values:
    print(f"{param}: {best_hps.values[param]}")


# Train the best model
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=10)