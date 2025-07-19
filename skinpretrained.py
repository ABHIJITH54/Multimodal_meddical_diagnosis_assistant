# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# train_dir = 'Dermnet/train'
# test_dir = 'Dermnet/test'
# img_size = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 50

# train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=25,width_shift_range=0.15,height_shift_range=0.15,shear_range=0.1,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(train_dir,target_size=img_size,batch_size=BATCH_SIZE,class_mode='sparse')

# test_generator = test_datagen.flow_from_directory(test_dir,target_size=img_size,batch_size=BATCH_SIZE,class_mode='sparse',shuffle=True)


# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False 

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.4)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.3)(x)
# predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
#     ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, verbose=1),
#     ModelCheckpoint('best_dermnet_mobilenetv2.keras', save_best_only=True)
# ]
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     validation_data=test_generator,
#     callbacks=callbacks
# )

# model.save("dermnet_mobilenetv2_final.keras")
# print("Model saved successfully!")

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()



import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# --- Config ---
train_dir = 'Dermnet/train'
test_dir = 'Dermnet/test'
img_size = (224, 224)
batch_size = 32
epochs = 50

# --- Data Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
)

# --- Model ---
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# --- Compile ---
model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_dermnet_effnetb0.keras', save_best_only=True, monitor='val_accuracy')
]

# --- Train ---
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=callbacks
)

# --- Save Final Model ---
model.save("dermnet_effnetb0_final.keras")
print("✅ Model saved successfully!")
