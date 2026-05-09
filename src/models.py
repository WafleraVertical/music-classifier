# ==============================================================================
# Author: Luis Eduardo Polaco
# Description: Model definitions for music genre classification.
#
# Implements two architectures for comparison:
#   - CNN: 4 convolutional blocks with GlobalAveragePooling2D for classification.
#   - CRNN: 3 convolutional blocks + Bidirectional GRU to capture temporal dependencies.
#
# Architecture: Input (128 × 1292 × 1) → Conv blocks → GAP/RNN → Dense → softmax (10 genres)
# ==============================================================================

import logging

import tensorflow as tf
from tensorflow.keras import layers

from config import CRNNConfig, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


# ===================================================================
# SECTION 1: Reusable building blocks
# ===================================================================


def conv_block(
    input_tensor: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    pool_size: tuple[int, int] = (2, 2),
    use_batch_norm: bool = True,
) -> tf.Tensor:
    """
    Reusable convolutional block: Conv2D → [BatchNorm] → ReLU → MaxPool.

    Args:
        input_tensor: Input tensor with shape (batch, height, width, channels).
        filters: Number of Conv2D filters (e.g. 32, 64, 128, 256).
        kernel_size: Convolutional filter size. Default (3, 3).
        pool_size: MaxPooling window size. Default (2, 2).
        use_batch_norm: If True, adds BatchNormalization before ReLU.

    Returns:
        Output tensor with height and width halved by MaxPool.
    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=not use_batch_norm,  # BN already has a bias-equivalent (beta)
    )(input_tensor)

    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)

    return x


# ===================================================================
# SECTION 2: CNN model
# ===================================================================


def build_cnn_model(config: ModelConfig) -> tf.keras.Model:
    """
    Build the pure CNN model using the Keras Functional API.

    Applies sequential convolutional blocks with growing filter counts,
    followed by GlobalAveragePooling2D and Dense layers for classification.

    Args:
        config: ModelConfig instance with all hyperparameters.
                Must have input_shape and num_classes set before calling
                (via setup_input_shapes and setup_num_classes in ProjectConfig).

    Returns:
        Uncompiled Keras model. Call compile_model() afterwards.

    Raises:
        ValueError: If input_shape or num_classes are not configured.
    """
    if config.input_shape is None:
        raise ValueError(
            "config.input_shape is not set. "
            "Call project_config.setup_input_shapes() first."
        )
    if config.num_classes is None:
        raise ValueError(
            "config.num_classes is not set. "
            "Call project_config.setup_num_classes(n) first."
        )

    inputs = tf.keras.Input(shape=config.input_shape, name="spectrogram_input")

    x = inputs
    for i, block_cfg in enumerate(config.conv_blocks):
        x = conv_block(
            input_tensor=x,
            filters=block_cfg.filters,
            kernel_size=block_cfg.kernel_size,
            pool_size=block_cfg.pool_size,
            use_batch_norm=block_cfg.use_batch_norm,
        )
        logger.debug(
            f"Conv block {i + 1}: filters={block_cfg.filters}, "
            f"output_shape={x.shape}"
        )

    # --- Global pooling or Flatten ---
    if config.use_global_avg_pooling:
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    # --- Dense layers with Dropout ---
    for i, units in enumerate(config.dense_units):
        x = layers.Dense(units, activation=config.activation, name=f"dense_{i + 1}")(x)
        x = layers.Dropout(config.dropout_rate, name=f"dropout_{i + 1}")(x)

    # --- Output layer ---
    outputs = layers.Dense(
        config.num_classes,
        activation=config.output_activation,
        name="output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_genre_classifier",
    )

    logger.info(
        f"CNN built — blocks={len(config.conv_blocks)}, "
        f"dense={config.dense_units}, "
        f"params={model.count_params():,}"
    )

    return model


# ===================================================================
# SECTION 3: CRNN model (counterproposal)
# ===================================================================


def build_crnn_model(config: CRNNConfig) -> tf.keras.Model:
    """
    Build the CRNN model using the Keras Functional API.

    Architecture: CNN blocks for spatial feature extraction,
    followed by a Reshape to convert the 2D feature maps into
    a temporal sequence, then a recurrent layer (GRU or LSTM)
    to capture long-range temporal dependencies.

    The reshape step is the key difference from the pure CNN:
        CNN output:    (batch, freq_reduced, time_reduced, filters)
        After Permute: (batch, time_reduced, freq_reduced, filters)
        After Reshape: (batch, time_reduced, freq_reduced * filters)
    Each time column of the spectrogram becomes one timestep
    with all its frequency information packed as features.

    Args:
        config: CRNNConfig instance with all hyperparameters.
                Must have input_shape and num_classes set before calling.

    Returns:
        Uncompiled Keras model. Call compile_model() afterwards.

    Raises:
        ValueError: If input_shape or num_classes are not configured.
        ValueError: If rnn_type is not 'gru' or 'lstm'.
    """
    if config.input_shape is None:
        raise ValueError(
            "config.input_shape is not set. "
            "Call project_config.setup_input_shapes() first."
        )
    if config.num_classes is None:
        raise ValueError(
            "config.num_classes is not set. "
            "Call project_config.setup_num_classes(n) first."
        )
    if config.rnn_type not in ("gru", "lstm"):
        raise ValueError(
            f"Invalid rnn_type '{config.rnn_type}'. Must be 'gru' or 'lstm'."
        )

    # --- Input ---
    inputs = tf.keras.Input(shape=config.input_shape, name="spectrogram_input")

    # --- CNN blocks ---
    x = inputs
    for i, block_cfg in enumerate(config.conv_blocks):
        x = conv_block(
            input_tensor=x,
            filters=block_cfg.filters,
            kernel_size=block_cfg.kernel_size,
            pool_size=block_cfg.pool_size,
            use_batch_norm=block_cfg.use_batch_norm,
        )
        logger.debug(
            f"CRNN conv block {i + 1}: filters={block_cfg.filters}, "
            f"output_shape={x.shape}"
        )

    # --- Reshape: (batch, freq, time, filters) → (batch, time, freq * filters) ---
    # Permute moves the time axis (2) to position 1 so the RNN
    # treats each time frame as one step in the sequence.
    x = layers.Permute((2, 1, 3), name="permute_time_first")(x)

    # Merge freq and filters into a single feature vector per timestep.
    _, time_steps, freq_reduced, last_filters = x.shape
    x = layers.Reshape(
        (time_steps, freq_reduced * last_filters),
        name="reshape_for_rnn",
    )(x)

    logger.debug(f"CRNN shape after reshape: {x.shape}")

    # --- Recurrent layer ---
    rnn_layer = layers.GRU if config.rnn_type == "gru" else layers.LSTM
    rnn = rnn_layer(
        units=config.rnn_units,
        dropout=config.rnn_dropout,
        return_sequences=False,
        name=f"{config.rnn_type}_layer",
    )

    if config.rnn_bidirectional:
        x = layers.Bidirectional(rnn, name=f"bidirectional_{config.rnn_type}")(x)
    else:
        x = rnn(x)

    # --- Dense layers with Dropout ---
    for i, units in enumerate(config.dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = layers.Dropout(config.dropout_rate, name=f"dropout_{i + 1}")(x)

    # --- Output layer ---
    outputs = layers.Dense(
        config.num_classes,
        activation="softmax",
        name="output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="crnn_genre_classifier",
    )

    logger.info(
        f"CRNN built — cnn_blocks={len(config.conv_blocks)}, "
        f"rnn={config.rnn_type}({'bi' if config.rnn_bidirectional else 'uni'})"
        f"x{config.rnn_units}, "
        f"params={model.count_params():,}"
    )

    return model


# ===================================================================
# SECTION 4: Model utilities
# ===================================================================


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Return the model summary as a string for logging and documentation.

    Args:
        model: A built (not necessarily compiled) Keras model.

    Returns:
        String containing the full model summary table.
    """
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)


def count_parameters(model: tf.keras.Model) -> dict[str, int]:
    """
    Count total, trainable, and non-trainable parameters.

    Useful for the thesis model comparison table.

    Args:
        model: A built Keras model.

    Returns:
        Dictionary with keys 'total', 'trainable', 'non_trainable'.
    """
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    non_trainable = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights))

    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def compile_model(
    model: tf.keras.Model,
    config: TrainingConfig,
) -> tf.keras.Model:
    """
    Compile a Keras model with optimizer, loss, and metrics from TrainingConfig.

    F1-Score is computed via a custom callback during training (see training.py),
    so here we only attach the built-in metrics: accuracy and loss tracking.
    The primary metric for early stopping and checkpointing is val_loss,
    which correlates well with F1 during training.

    Args:
        model: Uncompiled Keras model returned by build_cnn_model
               or build_crnn_model.
        config: TrainingConfig instance with optimizer and learning rate.

    Returns:
        The same model, now compiled and ready for model.fit().

    Raises:
        ValueError: If config.optimizer is not 'adam'.
    """
    if config.optimizer != "adam":
        raise ValueError(
            f"Optimizer '{config.optimizer}' is not supported yet. Use 'adam'."
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=config.loss_function,
        metrics=["accuracy"],
    )

    logger.info(
        f"Model compiled — optimizer=Adam(lr={config.learning_rate}), "
        f"loss={config.loss_function}"
    )

    return model
