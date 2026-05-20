from classification_models_3D.kkeras import Classifiers
import numpy as np
import random, os, re, pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import backend as K
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling3D
from keras.models import Model
from keras.src.utils import summary_utils
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate
use_weights = 'imagenet'
shape_size = (96, 128, 128, 3)
backbone = 'resnet18'
num_classes = 2
batch_size_train = 12
batch_size_valid = 12
learning_rate = 0.0001
patience = 10
epochs = 50
steps_per_epoch = 100
validation_steps = 20
dropout_val = 0.1
labels = pd.read_csv("/csl/users/2026lzhu/LABELS.csv").set_index("Subject")
labels = labels[~labels.index.duplicated(keep='first')]
random.seed(42)
np.random.seed(42)

models, answers = [], []
for file in os.listdir("/csl/users/2026lzhu/3D_VOLUMES"):
    subjnum = re.search("^\d+", file).group()
    models.append(np.load(file))
    answers.append({"Control":0, "Prodromal":1, "PD":2}[labels.loc[subjnum]["Group"]])
    
X_train, X_val, y_train, y_val = train_test_split(models, answers, test_size=0.25, random_state=42, stratify=answers)

def augment(volume, max_rotation=10):
    if random.random() < 0.5:
        angle = np.random.uniform(-max_rotation, max_rotation)
        volume = rotate(volume, angle, axes=(1, 2), reshape=False, order=1, mode='nearest')
    if random.random() < 0.5:
        volume *= np.random.uniform(0.9, 1.1)
    if random.random() < 0.5:
        volume += np.random.normal(0, 0.01, volume.shape)
    volume = (volume - np.mean(volume)) / (np.std(volume)+ 1e-8)
    return volume

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model' or layer_type == 'Functional':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output.shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = summary_utils.count_params(model.trainable_weights)
    non_trainable_count = summary_utils.count_params(model.non_trainable_weights)

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def batch_generator(batch_size, models, answers, preprocess_input):
    iterator = 0
    while True:
        image_list = []
        answ_list = []
        if iterator+batch_size < len(models):
            for i in range(iterator, iterator+batch_size):
                img, answ = augment(models[i]), answers[i]
                image_list.append(img); answ_list.append(answ)
        else:
            for i in range(iterator, len(models)):
                img, answ = augment(models[i]), answers[i]
                image_list.append(img); answ_list.append(answ)
            iterator = 0

        image_list = np.array(image_list, dtype=np.float32)
        image_list = preprocess_input(image_list)
        answ_list = np.array(answ_list, dtype=np.float32)
        # print(image_list.shape, answ_list.shape)
        yield image_list, answ_list

modelPoint, preprocess_input = Classifiers.get(backbone)
model = modelPoint(
    input_shape=shape_size,
    include_top=False,
    weights=use_weights,
)
x = model.layers[-1].output
x = GlobalAveragePooling3D()(x)
x = Dropout(dropout_val)(x)
x = Dense(num_classes, name='prediction')(x)
x = Activation('sigmoid')(x)
model = Model(inputs=model.inputs, outputs=x)

print(model.summary())
print(get_model_memory_usage(batch_size_train, model))
optim = Adam(learning_rate=learning_rate)

loss_to_use = 'binary_crossentropy'
model.compile(optimizer=optim, loss=loss_to_use, metrics=['acc',])

cache_model_path = f"3d_cnn_temp_model.keras"
best_model_path = f"best_3d_cnn_model.keras"
callbacks = [
    ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=False, verbose=0),
    ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
    CSVLogger('history_{}_lr_{}.csv'.format(backbone, learning_rate), append=True),
    EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='min'),
]

gen_train = batch_generator(
    batch_size_train,
    X_train, y_train,
    preprocess_input
)
gen_valid = batch_generator(
    batch_size_valid,
    X_val, y_val,
    preprocess_input
)

history = model.fit(
    gen_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=gen_valid,
    validation_steps=validation_steps,
    verbose=1,
    initial_epoch=0,
    callbacks=callbacks
)

best_loss = max(history.history['val_loss'])
print('Training finished. Loss: {}'.format(best_loss))
