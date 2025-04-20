from typing import Optional

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Layer
import tensorflow as tf


def gradcam(weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """
    Generate a localization map (heatmap) using Gradient-weighted Class Activation Mapping
    (Grad-CAM) (https://arxiv.org/pdf/1610.02391.pdf).

    The values for the parameters can be obtained from
    :func:`eli5.keras.gradcam.gradcam_backend`.

    This function computes a weighted combination of activation maps and applies a ReLU,
    then normalizes the result to [0, 1].

    Parameters
    ----------
    weights : numpy.ndarray
        1D array of channel weights (α_k), typically obtained by global average pooling of gradients.
    activations : numpy.ndarray
        3D activation maps A^k with shape (H, W, C) from the target convolutional layer.

    Returns
    -------
    lmap : numpy.ndarray
        2D localization map of shape (H, W), with values in [0, 1].

    Notes
    -----
    Assumptions:
      * Input is an image and model output is class scores or probabilities.

    Credits
    -------
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


def gradcam_backend(
    model: Model,
    doc: np.ndarray,
    targets: Optional[list[int]],
    activation_layer: Layer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    Compute and return the components needed for Grad-CAM visualization.

    This function evaluates the activation maps, computes gradients of the
    target class score with respect to those activations, and derives channel
    weights by global average pooling of the gradients:

        α^c_k = mean_{i,j}( ∂y^c / ∂A^k_{ij} )

    The final Grad-CAM localization map is given by:

        L^c_{Grad-CAM} = ReLU( Σ_k α^c_k A^k )

    Parameters
    ----------
    model : keras.models.Model
        Differentiable image classification model.
    doc : numpy.ndarray
        Input array with batch dimension (shape (1, H, W, C)).
    targets : list of int or None
        List of length one specifying the class index to explain. If None,
        the top predicted class is used.
    activation_layer : keras.layers.Layer
        Convolutional layer whose output activations contribute to Grad-CAM.

    See :func:`eli5.keras.explain_prediction` for description of the
    ``model``, ``doc``, ``targets`` parameters.

    Returns
    -------
    weights : numpy.ndarray
        Array of channel weights α^c_k (shape (C,)).
    activations : numpy.ndarray
        Activation maps A^k (shape (H, W, C)).
    grads : numpy.ndarray
        Gradients ∂y^c / ∂A^k (shape (H, W, C)).
    predicted_idx : int
        Class index used for computing the gradients.
    predicted_val : float
        Model score y^c for the target class.
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


def _validate_target(target: int, output_shape: tuple) -> None:
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
