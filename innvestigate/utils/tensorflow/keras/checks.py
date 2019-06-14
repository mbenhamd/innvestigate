# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import inspect
import keras.engine.topology
import tensorflow.keras.layers
import tensorflow.keras.layers.advanced_activations
import tensorflow.keras.layers.convolutional
import tensorflow.keras.layers.convolutional_recurrent
import tensorflow.keras.layers.core
import tensorflow.keras.layers.cudnn_recurrent
import tensorflow.keras.layers.embeddings
import tensorflow.keras.layers.local
import tensorflow.keras.layers.noise
import tensorflow.keras.layers.normalization
import tensorflow.keras.layers.pooling
import tensorflow.keras.layers.recurrent
import tensorflow.keras.layers.wrappers
import tensorflow.keras.legacy.layers


# Prevents circular imports.
def get_kgraph():
    from . import graph as kgraph
    return kgraph


__all__ = [
    "get_current_layers",
    "get_known_layers",
    "get_activation_search_safe_layers",

    "contains_activation",
    "contains_kernel",
    "only_relu_activation",
    "is_network",
    "is_convnet_layer",
    "is_relu_convnet_layer",
    "is_average_pooling",
    "is_max_pooling",
    "is_input_layer",
    "is_batch_normalization_layer",
]


###############################################################################
###############################################################################
###############################################################################


def get_current_layers():
    """
    Returns a list of currently available layers in Keras.
    """
    class_set = set([(getattr(tensorflow.keras.layers, name), name)
                     for name in dir(tensorflow.keras.layers)
                     if (inspect.isclass(getattr(tensorflow.keras.layers, name)) and
                         issubclass(getattr(tensorflow.keras.layers, name),
                                    tensorflow.keras.engine.topology.Layer))])
    return [x[1] for x in sorted((str(x[0]), x[1]) for x in class_set)]


def get_known_layers():
    """
    Returns a list of tensorflow.keras layer we are aware of.
    """

    # Inside function to not break import if Keras changes.
    KNOWN_LAYERS = (
        keras.engine.topology.InputLayer,
        tensorflow.keras.layers.advanced_activations.ELU,
        tensorflow.keras.layers.advanced_activations.LeakyReLU,
        tensorflow.keras.layers.advanced_activations.PReLU,
        tensorflow.keras.layers.advanced_activations.Softmax,
        tensorflow.keras.layers.advanced_activations.ThresholdedReLU,
        tensorflow.keras.layers.convolutional.Conv1D,
        tensorflow.keras.layers.convolutional.Conv2D,
        tensorflow.keras.layers.convolutional.Conv2DTranspose,
        tensorflow.keras.layers.convolutional.Conv3D,
        tensorflow.keras.layers.convolutional.Conv3DTranspose,
        tensorflow.keras.layers.convolutional.Cropping1D,
        tensorflow.keras.layers.convolutional.Cropping2D,
        tensorflow.keras.layers.convolutional.Cropping3D,
        tensorflow.keras.layers.convolutional.SeparableConv1D,
        tensorflow.keras.layers.convolutional.SeparableConv2D,
        tensorflow.keras.layers.convolutional.UpSampling1D,
        tensorflow.keras.layers.convolutional.UpSampling2D,
        tensorflow.keras.layers.convolutional.UpSampling3D,
        tensorflow.keras.layers.convolutional.ZeroPadding1D,
        tensorflow.keras.layers.convolutional.ZeroPadding2D,
        tensorflow.keras.layers.convolutional.ZeroPadding3D,
        tensorflow.keras.layers.convolutional_recurrent.ConvLSTM2D,
        tensorflow.keras.layers.convolutional_recurrent.ConvRecurrent2D,
        tensorflow.keras.layers.core.Activation,
        tensorflow.keras.layers.core.ActivityRegularization,
        tensorflow.keras.layers.core.Dense,
        tensorflow.keras.layers.core.Dropout,
        tensorflow.keras.layers.core.Flatten,
        tensorflow.keras.layers.core.Lambda,
        tensorflow.keras.layers.core.Masking,
        tensorflow.keras.layers.core.Permute,
        tensorflow.keras.layers.core.RepeatVector,
        tensorflow.keras.layers.core.Reshape,
        tensorflow.keras.layers.core.SpatialDropout1D,
        tensorflow.keras.layers.core.SpatialDropout2D,
        tensorflow.keras.layers.core.SpatialDropout3D,
        tensorflow.keras.layers.cudnn_recurrent.CuDNNGRU,
        tensorflow.keras.layers.cudnn_recurrent.CuDNNLSTM,
        tensorflow.keras.layers.embeddings.Embedding,
        tensorflow.keras.layers.local.LocallyConnected1D,
        tensorflow.keras.layers.local.LocallyConnected2D,
        tensorflow.keras.layers.Add,
        tensorflow.keras.layers.Average,
        tensorflow.keras.layers.Concatenate,
        tensorflow.keras.layers.Dot,
        tensorflow.keras.layers.Maximum,
        tensorflow.keras.layers.Minimum,
        tensorflow.keras.layers.Multiply,
        tensorflow.keras.layers.Subtract,
        tensorflow.keras.layers.noise.AlphaDropout,
        tensorflow.keras.layers.noise.GaussianDropout,
        tensorflow.keras.layers.noise.GaussianNoise,
        tensorflow.keras.layers.normalization.BatchNormalization,
        tensorflow.keras.layers.pooling.AveragePooling1D,
        tensorflow.keras.layers.pooling.AveragePooling2D,
        tensorflow.keras.layers.pooling.AveragePooling3D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling1D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling2D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling3D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling1D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling2D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling3D,
        tensorflow.keras.layers.pooling.MaxPooling1D,
        tensorflow.keras.layers.pooling.MaxPooling2D,
        tensorflow.keras.layers.pooling.MaxPooling3D,
        tensorflow.keras.layers.recurrent.GRU,
        tensorflow.keras.layers.recurrent.GRUCell,
        tensorflow.keras.layers.recurrent.LSTM,
        tensorflow.keras.layers.recurrent.LSTMCell,
        tensorflow.keras.layers.recurrent.RNN,
        tensorflow.keras.layers.recurrent.SimpleRNN,
        tensorflow.keras.layers.recurrent.SimpleRNNCell,
        tensorflow.keras.layers.recurrent.StackedRNNCells,
        tensorflow.keras.layers.wrappers.Bidirectional,
        tensorflow.keras.layers.wrappers.TimeDistributed,
        tensorflow.keras.layers.wrappers.Wrapper,
        tensorflow.keras.legacy.layers.Highway,
        tensorflow.keras.legacy.layers.MaxoutDense,
        tensorflow.keras.legacy.layers.Merge,
        tensorflow.keras.legacy.layers.Recurrent,
    )
    return KNOWN_LAYERS


def get_activation_search_safe_layers():
    """
    Returns a list of tensorflow.keras layer that we can walk along
    in an activation search.
    """

    # Inside function to not break import if Keras changes.
    ACTIVATION_SEARCH_SAFE_LAYERS = (
        tensorflow.keras.layers.advanced_activations.ELU,
        tensorflow.keras.layers.advanced_activations.LeakyReLU,
        tensorflow.keras.layers.advanced_activations.PReLU,
        tensorflow.keras.layers.advanced_activations.Softmax,
        tensorflow.keras.layers.advanced_activations.ThresholdedReLU,
        tensorflow.keras.layers.core.Activation,
        tensorflow.keras.layers.core.ActivityRegularization,
        tensorflow.keras.layers.core.Dropout,
        tensorflow.keras.layers.core.Flatten,
        tensorflow.keras.layers.core.Reshape,
        tensorflow.keras.layers.Add,
        tensorflow.keras.layers.noise.GaussianNoise,
        tensorflow.keras.layers.normalization.BatchNormalization,
    )
    return ACTIVATION_SEARCH_SAFE_LAYERS


###############################################################################
###############################################################################
###############################################################################


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "activation"):
        if activation is not None:
            return layer.activation == tensorflow.keras.activations.get(activation)
        else:
            return True
    elif isinstance(layer, tensorflow.keras.layers.ReLU):
        if activation is not None:
            return (tensorflow.keras.activations.get("relu") ==
                    tensorflow.keras.activations.get(activation))
        else:
            return True
    elif isinstance(layer, (
            tensorflow.keras.layers.advanced_activations.ELU,
            tensorflow.keras.layers.advanced_activations.LeakyReLU,
            tensorflow.keras.layers.advanced_activations.PReLU,
            tensorflow.keras.layers.advanced_activations.Softmax,
            tensorflow.keras.layers.advanced_activations.ThresholdedReLU)):
        if activation is not None:
            raise Exception("Cannot detect activation type.")
        else:
            return True
    else:
        return False


def contains_kernel(layer):
    """
    Check whether the layer contains a kernel.
    """

    # TODO: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "kernel") or hasattr(layer, "depthwise_kernel") or hasattr(layer, "pointwise_kernel"):
        return True
    else:
        return False


def contains_bias(layer):
    """
    Check whether the layer contains a bias.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "bias"):
        return True
    else:
        return False


def only_relu_activation(layer):
    """Checks if layer contains no or only a ReLU activation."""
    return (not contains_activation(layer) or
            contains_activation(layer, None) or
            contains_activation(layer, "linear") or
            contains_activation(layer, "relu"))


def is_network(layer):
    """
    Is network in network?
    """
    return isinstance(layer, tensorflow.keras.engine.topology.Network)


def is_conv_layer(layer, *args, **kwargs):
    """Checks if layer is a convolutional layer."""
    CONV_LAYERS = (
        tensorflow.keras.layers.convolutional.Conv1D,
        tensorflow.keras.layers.convolutional.Conv2D,
        tensorflow.keras.layers.convolutional.Conv2DTranspose,
        tensorflow.keras.layers.convolutional.Conv3D,
        tensorflow.keras.layers.convolutional.Conv3DTranspose,
        tensorflow.keras.layers.convolutional.SeparableConv1D,
        tensorflow.keras.layers.convolutional.SeparableConv2D,
        tensorflow.keras.layers.convolutional.DepthwiseConv2D
    )
    return isinstance(layer, CONV_LAYERS)


def is_batch_normalization_layer(layer, *args, **kwargs):
    """Checks if layer is a batchnorm layer."""
    return isinstance(layer, tensorflow.keras.layers.normalization.BatchNormalization)


def is_add_layer(layer, *args, **kwargs):
    """Checks if layer is an addition-merge layer."""
    return isinstance(layer, tensorflow.keras.layers.Add)


def is_dense_layer(layer, *args, **kwargs):
    """Checks if layer is a dense layer."""
    return isinstance(layer, tensorflow.keras.layers.core.Dense)


def is_convnet_layer(layer):
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.
    CONVNET_LAYERS = (
        keras.engine.topology.InputLayer,
        tensorflow.keras.layers.advanced_activations.ELU,
        tensorflow.keras.layers.advanced_activations.LeakyReLU,
        tensorflow.keras.layers.advanced_activations.PReLU,
        tensorflow.keras.layers.advanced_activations.Softmax,
        tensorflow.keras.layers.advanced_activations.ThresholdedReLU,
        tensorflow.keras.layers.convolutional.Conv1D,
        tensorflow.keras.layers.convolutional.Conv2D,
        tensorflow.keras.layers.convolutional.Conv2DTranspose,
        tensorflow.keras.layers.convolutional.Conv3D,
        tensorflow.keras.layers.convolutional.Conv3DTranspose,
        tensorflow.keras.layers.convolutional.Cropping1D,
        tensorflow.keras.layers.convolutional.Cropping2D,
        tensorflow.keras.layers.convolutional.Cropping3D,
        tensorflow.keras.layers.convolutional.SeparableConv1D,
        tensorflow.keras.layers.convolutional.SeparableConv2D,
        tensorflow.keras.layers.convolutional.UpSampling1D,
        tensorflow.keras.layers.convolutional.UpSampling2D,
        tensorflow.keras.layers.convolutional.UpSampling3D,
        tensorflow.keras.layers.convolutional.ZeroPadding1D,
        tensorflow.keras.layers.convolutional.ZeroPadding2D,
        tensorflow.keras.layers.convolutional.ZeroPadding3D,
        tensorflow.keras.layers.core.Activation,
        tensorflow.keras.layers.core.ActivityRegularization,
        tensorflow.keras.layers.core.Dense,
        tensorflow.keras.layers.core.Dropout,
        tensorflow.keras.layers.core.Flatten,
        tensorflow.keras.layers.core.Lambda,
        tensorflow.keras.layers.core.Masking,
        tensorflow.keras.layers.core.Permute,
        tensorflow.keras.layers.core.RepeatVector,
        tensorflow.keras.layers.core.Reshape,
        tensorflow.keras.layers.core.SpatialDropout1D,
        tensorflow.keras.layers.core.SpatialDropout2D,
        tensorflow.keras.layers.core.SpatialDropout3D,
        tensorflow.keras.layers.embeddings.Embedding,
        tensorflow.keras.layers.local.LocallyConnected1D,
        tensorflow.keras.layers.local.LocallyConnected2D,
        tensorflow.keras.layers.Add,
        tensorflow.keras.layers.Average,
        tensorflow.keras.layers.Concatenate,
        tensorflow.keras.layers.Dot,
        tensorflow.keras.layers.Maximum,
        tensorflow.keras.layers.Minimum,
        tensorflow.keras.layers.Multiply,
        tensorflow.keras.layers.Subtract,
        tensorflow.keras.layers.noise.AlphaDropout,
        tensorflow.keras.layers.noise.GaussianDropout,
        tensorflow.keras.layers.noise.GaussianNoise,
        tensorflow.keras.layers.normalization.BatchNormalization,
        tensorflow.keras.layers.pooling.AveragePooling1D,
        tensorflow.keras.layers.pooling.AveragePooling2D,
        tensorflow.keras.layers.pooling.AveragePooling3D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling1D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling2D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling3D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling1D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling2D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling3D,
        tensorflow.keras.layers.pooling.MaxPooling1D,
        tensorflow.keras.layers.pooling.MaxPooling2D,
        tensorflow.keras.layers.pooling.MaxPooling3D,
    )
    return isinstance(layer, CONVNET_LAYERS)


def is_relu_convnet_layer(layer):
    """Checks if layer is from a convolutional network with ReLUs."""
    return (is_convnet_layer(layer) and only_relu_activation(layer))


def is_average_pooling(layer):
    """Checks if layer is an average-pooling layer."""
    AVERAGEPOOLING_LAYERS = (
        tensorflow.keras.layers.pooling.AveragePooling1D,
        tensorflow.keras.layers.pooling.AveragePooling2D,
        tensorflow.keras.layers.pooling.AveragePooling3D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling1D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling2D,
        tensorflow.keras.layers.pooling.GlobalAveragePooling3D,
    )
    return isinstance(layer, AVERAGEPOOLING_LAYERS)


def is_max_pooling(layer):
    """Checks if layer is a max-pooling layer."""
    MAXPOOLING_LAYERS = (
        tensorflow.keras.layers.pooling.MaxPooling1D,
        tensorflow.keras.layers.pooling.MaxPooling2D,
        tensorflow.keras.layers.pooling.MaxPooling3D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling1D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling2D,
        tensorflow.keras.layers.pooling.GlobalMaxPooling3D,
    )
    return isinstance(layer, MAXPOOLING_LAYERS)


def is_input_layer(layer, ignore_reshape_layers=True):
    """Checks if layer is an input layer."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.
    kgraph = get_kgraph()

    layer_inputs = kgraph.get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        tensorflow.keras.layers.Flatten,
        tensorflow.keras.layers.Permute,
        tensorflow.keras.layers.Reshape,
    )
    while any([isinstance(x, IGNORED_LAYERS) for x in layer_inputs]):
        tmp = set()
        for l in layer_inputs:
            if(ignore_reshape_layers and
               isinstance(l, IGNORED_LAYERS)):
                tmp.update(kgraph.get_input_layers(l))
            else:
                tmp.add(l)
        layer_inputs = tmp

    if all([isinstance(x, tensorflow.keras.layers.InputLayer)
            for x in layer_inputs]):
        return True
    else:
        return False
