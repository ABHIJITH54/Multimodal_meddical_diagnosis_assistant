
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

train_dir = 'SkinDisease/train'
test_dir = 'SkinDisease/test'
image_size = (224, 224)
batch_size = 32
initial_epochs = 20
fine_tune_epochs = 50
total_epochs = initial_epochs + fine_tune_epochs

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))


with open("class_indices_resnet101.json", "w") as f:
    json.dump(train_generator.class_indices, f)


base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
base_model.trainable = False  

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss_fn, metrics=['accuracy'])


early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)
checkpoint = callbacks.ModelCheckpoint("best_resnet101_skin.keras", monitor="val_accuracy", save_best_only=True)

history_frozen = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

base_model.trainable = True
for layer in base_model.layers[:100]: 
    layer.trainable = False


model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=['accuracy'])


history_finetune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history_frozen.epoch[-1] + 1,
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)


loss, acc = model.evaluate(test_generator)
print(f"Final Test Accuracy (ResNet101): {acc:.4f}")

model.save("final_resnet101_skin.keras")
