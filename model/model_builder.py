"""
model_builder.py
Defines the CNN architectures used to classify medicinal plant leaves.

Two options are provided:
  - build_cnn():            a small from-scratch CNN (4 conv blocks).
                             Simple and fast, but needs LOTS of images
                             per class (100+) to reach good accuracy.
  - build_transfer_model(): MobileNetV2 pretrained on ImageNet, with a
                             small classifier head on top. Recommended
                             default — reaches much higher accuracy with
                             far fewer training images (20-50/class is
                             often enough), since it starts from a model
                             that already understands general visual
                             features (edges, textures, shapes) instead
                             of learning everything from zero.
"""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_transfer_model(input_shape, num_classes, fine_tune=False, fine_tune_at=100):
    """
    Transfer-learning model: frozen (or partially fine-tuned) MobileNetV2
    backbone + a small trainable classifier head.

    Expects input already scaled to [0, 1] — exactly what
    utils.image_utils.preprocess_for_model() and
    ImageDataGenerator(rescale=1./255) both produce. A Lambda layer here
    rescales internally to the [-1, 1] range MobileNetV2 was trained on,
    so nothing elsewhere in the pipeline needs to change.

    fine_tune=False (default): backbone stays frozen — fastest, safest
        choice for small datasets (tens of images per class).
    fine_tune=True: unfreezes layers from `fine_tune_at` onward for a
        light fine-tune pass — can help once you have 100+ images/class,
        but risks overfitting on very small datasets.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda img: img * 2.0 - 1.0,
                       name="rescale_0_1_to_minus1_1")(inputs)

    base_model = MobileNetV2(input_shape=input_shape, include_top=False,
                              weights="imagenet")
    base_model.trainable = fine_tune
    if fine_tune:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="PlantAI_MobileNetV2")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape, num_classes):
    """
    A compact CNN:
      Conv(32) -> Conv(64) -> Conv(128) -> Conv(128) -> Dense(256) -> Softmax
    Includes BatchNorm + Dropout to reduce overfitting on small datasets.
    """
    model = models.Sequential(name="PlantAI_CNN")

    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model