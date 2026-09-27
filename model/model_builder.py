"""
model_builder.py
Defines the CNN architecture used to classify medicinal plant leaves.

UPDATED: now uses transfer learning (MobileNetV2 pretrained on ImageNet)
instead of a from-scratch CNN. A from-scratch CNN needs hundreds of images
per class to learn useful features; with small datasets (~20 images/class)
it just memorizes the training set and fails on new photos (val accuracy
stays near random chance while training accuracy climbs - classic
overfitting). MobileNetV2 already knows general visual features (edges,
textures, shapes) from 1.4M ImageNet images, so we only need to train a
small classifier head on top of it - this works far better with limited
data.

The base MobileNetV2 layers are frozen (not trained) so we don't destroy
their pretrained knowledge; only the new head layers are trained.

Note: the first time this runs, Keras will download the pretrained
ImageNet weights (~9-14 MB) automatically, so an internet connection is
required for that one-time download.
"""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_cnn(input_shape, num_classes):
    """
    Transfer-learning model:
      MobileNetV2 (frozen, ImageNet weights) -> GAP -> Dense(128) -> Dropout -> Softmax

    Input is expected in [0, 1] float range (matching image_utils.py's
    preprocess_for_model and train.py's ImageDataGenerator rescale=1/255),
    so we rescale to the [-1, 1] range MobileNetV2 expects internally.
    """
    inputs = layers.Input(shape=input_shape)

    # Map [0, 1] -> [-1, 1], the range MobileNetV2's ImageNet weights expect.
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze pretrained feature extractor

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="PlantAI_CNN")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model