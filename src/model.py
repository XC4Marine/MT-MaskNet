from tensorflow.keras import layers, models, Input
from tensorflow.keras.regularizers import l2
import keras.ops as ops  

def resnet_block(input_tensor, filters, strides=1):
    """Residual block for ResNet18.
    
    Args:
        input_tensor: Input tensor.
        filters: Number of filters.
        strides: Stride for convolution (default 1).
    
    Returns:
        Output tensor after residual block.
    """
    shortcut = input_tensor
    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=strides, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(filters, (3,3), strides=strides, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet18(input_shape, num_classes):
    """Build ResNet18 baseline model.
    
    Args:
        input_shape: Input shape (e.g., (128, 128, 1)).
        num_classes: Number of output classes.
    
    Returns:
        Compiled Keras model (not compiled here; done in train scripts).
    """
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(64, (7,7), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 256)
    x = resnet_block(x, 512, strides=2)
    x = resnet_block(x, 512)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model


def channel_minmax_normalize(inputs, epsilon=1e-7):
    """Normalize channels to [0,1]."""
    mins = ops.min(inputs, axis=[1, 2], keepdims=True)
    maxs = ops.max(inputs, axis=[1, 2], keepdims=True)
    return (inputs - mins) / (maxs - mins + epsilon)

def channel_minmax_invert(inputs, epsilon=1e-7):
    """Invert normalized channels."""
    return 1 - channel_minmax_normalize(inputs, epsilon)

def build_mt_masknet(sound_model_path, noise_model_path, relu_layer_name, bn_layer_name):
    """Build MT-MaskNet with configurable gating layers.
    
    Args:
        ... (previous)
        relu_layer_name: ReLU layer name from model.summary() (e.g., 're_lu_6' for Layer1).
        bn_layer_name: BatchNorm layer name (e.g., 'batch_normalization_9').
    
    Returns:
        MT-MaskNet model.
    """
    spec_model = tf.keras.models.load_model(sound_model_path)
    gate_model = tf.keras.models.load_model(noise_model_path)
    
    spec_feature = tf.keras.Model(
        inputs=spec_model.input,
        outputs=[
            spec_model.get_layer(relu_layer_name).output,
            spec_model.get_layer(bn_layer_name).output
        ]
    )
    
    gate_feature = tf.keras.Model(
        inputs=gate_model.input,
        outputs=[
            gate_model.get_layer(relu_layer_name).output,
            gate_model.get_layer(bn_layer_name).output
        ]
    )
    
    # Inputs
    input_spec = Input(shape=(128, 128, 1), name="spec_input")
    input_gate = Input(shape=(128, 128, 1), name="gate_input")
    
    # Extract features
    re_lu_51, spec_feat_63 = spec_feature(input_spec)
    re_lu_66, gate_feat_81 = gate_feature(input_gate)
    
    # Gating mechanism
    spec_feat = channel_minmax_normalize(spec_feat_63)
    gate_feat = channel_minmax_invert(gate_feat_81)
    
    x = layers.Multiply()([spec_feat, gate_feat])
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, re_lu_51])
    x = layers.ReLU()(x)
    
    # Additional ResNet blocks
    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 256)
    x = resnet_block(x, 512, strides=2)
    x = resnet_block(x, 512)
    
    # Voice output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = layers.Dropout(0.5)(x)
    voice_type_output = layers.Dense(3, activation='softmax', name='voice_type_output')(x)
    
    # Noise output
    y = gate_feat_81
    y = layers.BatchNormalization()(y)
    y = layers.Add()([y, re_lu_66])
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(y)
    y = layers.Dropout(0.5)(y)
    noise_type_output = layers.Dense(4, activation='softmax', name='noise_type_output')(y)
    
    model = models.Model(inputs=[input_spec, input_gate],
                         outputs=[voice_type_output, noise_type_output])
    return model
