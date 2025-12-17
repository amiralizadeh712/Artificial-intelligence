import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# -------------------------
# تنظیمات دیتاست
# -------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16   # می‌توانید CPU را سبک نگه دارید

# -------------------------
# آماده‌سازی داده‌ها
# -------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = train_data.num_classes
print("Number of classes:", num_classes)

# -------------------------
# تعریف AlexNet Base
# -------------------------
def alexnet_base(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Conv2D(96, (11,11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3,3), strides=2),

        layers.Conv2D(256, (5,5), padding='same', activation='relu'),
        layers.MaxPooling2D((3,3), strides=2),

        layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((3,3), strides=2),
    ])
    return model

base_model = alexnet_base()
for layer in base_model.layers:
    layer.trainable = False   # فریز کردن AlexNet

# -------------------------
# اضافه کردن لایه‌های Dense
# -------------------------
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# آموزش مدل
# -------------------------
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data
)

# -------------------------
# رسم نمودار Accuracy و Loss
# -------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------
# ذخیره بهترین Accuracy
# -------------------------
best_val_acc = max(history.history['val_accuracy'])
print("Best Validation Accuracy:", best_val_acc)
