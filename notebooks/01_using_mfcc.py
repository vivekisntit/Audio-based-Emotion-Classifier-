import os
import pandas as pd
import numpy as np
import librosa
from tensorflow.keras.callbacks import EarlyStopping

"""#### Setting path to RAVDESS

Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav

03-01-05-01-02-02-12.wav

we'll only keep 01, 03, 04, 05, 06, 07
"""

DATA_PATH = "../datasets/ravdess"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust"
}

file_paths = []
emotions = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]

            if emotion_code in emotion_map:
                file_paths.append(os.path.join(root, file))
                emotions.append(emotion_map[emotion_code])

# Create dataframe
df = pd.DataFrame({
    "path": file_paths,
    "emotion": emotions
})

df.head()

df["emotion"].value_counts()

print("Total samples:", len(df))

"""### Audio standardization"""

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, padding))

    return signal

"""### Extract MFCC"""

def extract_mfcc(signal, sr=SAMPLE_RATE, n_mfcc=40):
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.stack((mfcc, delta, delta2), axis=-1)

    return combined

sample_signal = load_audio(df["path"].iloc[0])
mfcc = extract_mfcc(sample_signal)

print("MFCC shape:", mfcc.shape)

"""### Encode labels"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["label"] = le.fit_transform(df["emotion"])

print(le.classes_)
df.head()

X = []
y = []

for index, row in df.iterrows():
    signal = load_audio(row["path"])
    mfcc = extract_mfcc(signal)

    X.append(mfcc)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("New feature shape:", X.shape)

"""### Train/Test split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

"""### Normalization"""

# compute mean and std per feature channel
mean = np.mean(X_train, axis=(0,1,2), keepdims=True)
std = np.std(X_train, axis=(0,1,2), keepdims=True)

X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

"""### One-Hot Encoding"""

from tensorflow.keras.utils import to_categorical

num_classes = len(np.unique(y))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

"""### Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

input_shape = X_train.shape[1:]

model = Sequential()

# Block 1
model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

# Block 3
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

"""#### Model compilation"""

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',      # watch validation loss
    patience=8,              # wait 8 epochs before stopping
    restore_best_weights=True
)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)

class_weights = dict(enumerate(class_weights))
print(class_weights)

"""### Training the model"""

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    # callbacks=[early_stop]
)