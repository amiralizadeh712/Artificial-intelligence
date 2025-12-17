import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# -------------------------
# تنظیمات
# -------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

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
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_data.num_classes
print("Number of classes:", num_classes)

# -------------------------
# بارگذاری VGG-16
# -------------------------
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# فریز کردن VGG-16
for layer in base_model.layers:
    layer.trainable = False

# -------------------------
# اضافه کردن Dense Layers
# -------------------------
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# آموزش
# -------------------------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# -------------------------
# رسم نمودارها
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

print("Best Validation Accuracy:",
      max(history.history["val_accuracy"]))
