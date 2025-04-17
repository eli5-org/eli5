# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Tuple, List

import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Layer
import tensorflow as tf


def gradcam(weights, activations):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Generate a localization map (heatmap) using Gradient-weighted Class Activation Mapping 
    (Grad-CAM) (https://arxiv.org/pdf/1610.02391.pdf).
    
    The values for the parameters can be obtained from
    :func:`eli5.keras.gradcam.gradcam_backend`.

    Parameters
    ----------
    weights : numpy.ndarray
        Activation weights, vector with one weight per map, 
        rank 1.

    activations : numpy.ndarray
        Forward activation map values, vector of matrices, 
        rank 3.
    
    Returns
    -------
    lmap : numpy.ndarray
        A Grad-CAM localization map,
        rank 2, with values normalized in the interval [0, 1].

    Notes
    -----
    We currently make two assumptions in this implementation
        * We are dealing with images as our input to ``model``.
        * We are doing a classification. ``model``'s output is a class scores or probabilities vector.

    Credits
        * Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam".
        * Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to Jacob's implementation.
        * Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis".
    """
    # For reusability, this function should only use numpy operations
    # Instead of backend library operations
    
    # Perform a weighted linear combination
    # we need to multiply (dim1, dim2, maps,) by (maps,) over the first two axes
    # and add each result to (dim1, dim2,) results array
    # there does not seem to be an easy way to do this:
    # see: https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
    spatial_shape = activations.shape[:2] # -> (dim1, dim2)
    lmap = np.zeros(spatial_shape, dtype=np.float64)
    # iterate through each activation map
    for i, w in enumerate(weights): 
        # weight * spatial map
        # add result to the entire localization map (NOT pixel by pixel)
        lmap += w * activations[..., i]

    lmap = np.maximum(lmap, 0) # ReLU

    # normalize lmap to [0, 1] ndarray
    # add eps to avoid division by zero in case lmap is 0's
    # this also means that lmap max will be slightly less than the 'true' max
    lmap = lmap / (np.max(lmap)+K.epsilon())
    return lmap


def gradcam_backend(model,  # type: Model
                    doc,  # type: np.ndarray
                    targets,  # type: Optional[List[int]]
                    activation_layer  # type: Layer
                    ):
    # type: (...) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]
    """
    Compute Grad-CAM outputs: weights, activation maps, gradients,
    predicted index and predicted value using TensorFlow eager execution.
    """
    # Prepare the input tensor
    doc_tensor = tf.convert_to_tensor(doc)
    # Ensure the model.output is defined (e.g., for Sequential models)
    _ = model(doc_tensor)
    # Model mapping inputs to activations and predictions
    # Build model mapping inputs to desired activation outputs and final predictions
    # Use model.outputs[0] to support Sequential models in Keras 3.x
    grad_model = Model(inputs=model.inputs,
                       outputs=[activation_layer.output, model.outputs[0]])
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass
        conv_outputs, predictions = grad_model(doc_tensor)
        # Watch activations for gradient calculation
        tape.watch(conv_outputs)
        # Determine target class index
        if targets is None:
            pred_index = tf.argmax(predictions[0])
        else:
            if not isinstance(targets, list):
                raise TypeError(f'Invalid targets: {targets}')
            if len(targets) != 1:
                raise ValueError(f'More than one prediction target is not supported: {targets}')
            target = targets[0]
            _validate_target(target, model.output_shape)
            pred_index = tf.constant(target, dtype=tf.int32)
        # Select the score for the target class
        class_channel = predictions[:, pred_index]
    # Compute gradients of the class score w.r.t. convolutional outputs
    grads = tape.gradient(class_channel, conv_outputs)
    # Global average pooling of gradients to obtain weights
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))[0]
    # Extract values for the first (and only) sample
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    # Convert to numpy arrays
    weights = pooled_grads.numpy()
    activations = conv_outputs.numpy()
    grads = grads.numpy()
    predicted_idx = int(pred_index.numpy())
    predicted_val = float(predictions[0, pred_index].numpy())
    return weights, activations, grads, predicted_idx, predicted_val


def _calc_gradient(ys, xs):
    # (K.variable, list) -> K.variable
    """
    Return the gradient of scalar ``ys`` with respect to each of list ``xs``,
    (must be singleton)
    and apply grad normalization.
    """
    # Differentiate ys (scalar) w.r.t. each variable in xs using TensorFlow v1 gradients
    try:
        grads_list = tf.compat.v1.gradients(ys, xs)
    except Exception as e:
        raise ValueError(f'Gradient calculation failed: {e}')
    # Expect a single gradient tensor
    if not grads_list or grads_list[0] is None:
        raise ValueError(f'Gradient calculation resulted in None values. ys: {ys}. xs: {xs}.')
    grads = grads_list[0]
    # Normalize gradients (L2)
    grads = tf.math.l2_normalize(grads)
    return grads


def _get_target_prediction(targets, model):
    # type: (Optional[list], Model) -> K.variable
    """
    Get a prediction ID based on ``targets``, 
    from the model ``model`` (with a rank 2 tensor for its final layer).
    Returns a rank 1 K.variable tensor.
    """
    if isinstance(targets, list):
        # take the first prediction from the list
        if len(targets) == 1:
            target = targets[0]
            _validate_target(target, model.output_shape)
            predicted_idx = K.constant([target], dtype='int64')
        else:
            raise ValueError('More than one prediction target '
                             'is currently not supported ' 
                             '(found a list that is not length 1): '
                             '{}'.format(targets))
    elif targets is None:
        predicted_idx = K.argmax(model.output, axis=-1)
    else:
        raise TypeError('Invalid argument "targets" (must be list or None): %s' % targets)
    return predicted_idx


def _validate_target(target, output_shape):
    # type: (int, tuple) -> None
    """
    Check whether ``target``, 
    an integer index into the model's output
    is valid for the given ``output_shape``.
    """
    if isinstance(target, int):
        output_nodes = output_shape[1:][0]
        if not (0 <= target < output_nodes):
            raise ValueError('Prediction target index is ' 
                             'outside the required range [0, {}). '
                             'Got {}'.format(output_nodes, target))
    else:
        raise TypeError('Prediction target must be int. '
                        'Got: {}'.format(target))
