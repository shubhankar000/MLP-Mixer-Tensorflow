#%%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import *
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from tensorflow.keras.models import Model
from mlp_utils import *

# Comment line below to run on GPU
# tf.config.set_visible_devices([], "GPU")

#%%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# normalize pixel values
x_train = x_train / 255
x_test = x_test / 255

# create validation data from test data
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# %%
def create_model():
    dims = 512
    k, s = (4, 4)  # (kernel, strides)
    depth = 4

    inputs = Input((32, 32, 3))

    x = tf.keras.Sequential(
        [
            RandomFlip(),
            RandomRotation(factor=0.03),
            RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )(inputs)

    x = CreatePatches(k, s)(x)
    x = PerPatchFullyConnected(dims)(x)

    for _ in range(depth):
        x = MLPBlock()(x)

    # x = Projection()(x)
    x = GlobalAveragePooling1D()(x)

    # classification layer
    x = GaussianDropout(0.9)(x)
    output = Dense(len(classes), activation="softmax", kernel_regularizer="l2")(x)

    return Model(inputs=inputs, outputs=output)


model = create_model()
print(model.summary())

# %%
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adadelta(1),
    metrics=["accuracy"],
)

b_s = 128

val_dataset = val_dataset.batch(b_s)

history = model.fit(
    x_train,
    y_train,
    validation_data=val_dataset,
    batch_size=b_s,
    epochs=80,
    verbose="auto",
)

# # visualize
acc = [0.0] + history.history["accuracy"]
val_acc = [0.0] + history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

# %%
random = np.random.randint(0, 9999, 1)

for i in random:
    pred = model.predict(x_test[i][np.newaxis, :])
    class_prob = tf.math.reduce_max(pred).numpy()
    predictions = classes[np.argmax(pred)]
    label = classes[int(y_test[i])]
    print(
        f"Actual Label: {label} || Predicted: {predictions} || Certainity: {round(class_prob * 100, 2)}%"
    )
    plt.imshow(x_test[i])
    plt.show()

# %%
# Plot confusion matrix
preds = np.argmax(model.predict(x_test), axis=1)
confuse = confusion_matrix(y_test, preds)
plot_confusion_matrix(confuse, figsize=(8, 8), class_names=classes, show_normed=True)
