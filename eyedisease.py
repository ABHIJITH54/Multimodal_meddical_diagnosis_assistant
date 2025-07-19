import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras import layers,models,callbacks,regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import numpy as np  
import json
import os

data_dir=r"E:\Multimodal_meddical_diagnosis_assistant\dataset"
image_size=(224,224)
batch_size=32
initial_epochs=20
fine_tune_epochs=50
total_epochs=initial_epochs+fine_tune_epochs

datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=15,zoom_range=0.2,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,fill_mode='nearest',validation_split=0.2)
train_generator=datagen.flow_from_directory(data_dir,target_size=image_size,batch_size=batch_size,class_mode='categorical',shuffle=True,subset='training')
val_generator=datagen.flow_from_directory(data_dir,target_size=image_size,batch_size=batch_size,class_mode='categorical',shuffle=False,subset='validation')

class_weights=compute_class_weight(class_weight='balanced',classes=np.unique(train_generator.classes),y=train_generator.classes)
class_weights=dict(enumerate(class_weights))

with open("class_indices_resnet101_eye.json","w") as f:
    json.dump(train_generator.class_indices,f)


base_model=ResNet101(weights='imagenet',include_top=False,input_shape=(*image_size,3))
base_model.trainable=False

model=models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(512,activation='relu',kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes,activation='softmax')
])
lossfn=CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=Adam(learning_rate=1e-3),loss=lossfn,metrics=['accuracy'])

early_stop=callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
reduce_lr=callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.3,patience=5,verbose=1)
checkpoint = callbacks.ModelCheckpoint("best_resnet101_eye.keras", monitor="val_accuracy", save_best_only=True)

history_frozen=model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

base_model.trainable=True
for layer in base_model.layers[:100]:
    layer.trainable = False


model.compile(optimizer=Adam(learning_rate=1e-5), loss=lossfn, metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history_frozen.epoch[-1] + 1,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

loss, acc = model.evaluate(val_generator)
print(f"Final Validation Accuracy (ResNet101): {acc:.4f}")

model.save("final_resnet101_eye.keras")
