# ======================================================
# model_resnet.py
# Defines build_resnet_model() for Traffic Sign Detection
# Supports ResNet-50, ResNet-101, and ResNet-152
# ======================================================

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_resnet_model(
    version=50,
    input_shape=(64, 64, 3),
    num_classes=43,
    learning_rate=1e-4,
    fine_tune=False,
    fine_tune_at=-30
):
    """
    Build a ResNet model for Traffic Sign Classification.

    Parameters
    ----------
    version : int
        ResNet version (choose 50, 101, or 152)
    input_shape : tuple
        Shape of input images (default: 64x64x3)
    num_classes : int
        Number of traffic sign classes
    learning_rate : float
        Learning rate for Adam optimizer
    fine_tune : bool
        If True, unfreezes deeper layers for fine-tuning
    fine_tune_at : int
        Number of layers (from the end) to unfreeze for fine-tuning
    """

    # Select the correct ResNet variant
    if version == 50:
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif version == 101:
        base_model = ResNet101(weights="imagenet", include_top=False, input_shape=input_shape)
    elif version == 152:
        base_model = ResNet152(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("❌ Supported versions: 50, 101, 152")

    # Freeze or partially unfreeze layers
    for layer in base_model.layers:
        layer.trainable = False

    if fine_tune:
        print(f"🔧 Fine-tuning enabled. Unfreezing last {abs(fine_tune_at)} layers...")
        for layer in base_model.layers[fine_tune_at:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)

    # Build final model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Optional: run a quick summary when executed directly
if __name__ == "__main__":
    model = build_resnet_model(version=50, input_shape=(64, 64, 3), num_classes=43)
    model.summary()
