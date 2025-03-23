.. _library-keras:

Keras
=====

Keras_ is "a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano". 

Keras can be used for many Machine Learning tasks, and it has support for both popular
and experimental neural network architectures.

Note: only TensorFlow 1.x is supported, recommended Keras version is 2.3.1 or earlier, and eli5 version 0.13 or earlier, as you can't install TensorFlow 1.x on Python 3.9+ which is required for eli5 0.14+

.. _Keras: https://keras.io/


.. _keras-explain-prediction:

explain_prediction
------------------

Currently ELI5 supports :func:`eli5.explain_prediction` for Keras image classifiers.
:func:`eli5.explain_prediction` explains image classifications through `Grad-CAM <https://arxiv.org/pdf/1610.02391.pdf>`_.

The returned :class:`eli5.base.Explanation` instance contains some important objects:

* ``image`` represents the image input into the model. A Pillow image.

* ``targets`` represents the explanation values for each target class (currently only 1 target is supported). A list of :class:`eli5.base.TargetExplanation` objects with the following attributes set:

    * ``heatmap``  a grayscale "localization map" (rank 2 (2D) numpy array, with float values in the interval `[0, 1]`). The numbers indicate how important the region in the image is for the target class (even if the target class was not the predicted class). Higher numbers mean that the region tends to increase the predicted value for a class. Lower numbers mean that the region has smaller effect on the predicted class score.

    * ``target`` the integer ID of the class (same as the argument to ``targets`` if one was passed, or the predicted class ID if no argument was passed).

    * ``score`` the output of the network for the predicted class.


Important arguments to :func:`eli5.explain_prediction` for ``Model`` and ``Sequential``:

* ``doc`` is an image as a tensor that can be inputted to the model.
    
    - The tensor must be an instance of ``numpy.ndarray``.

    - Usually the tensor has the format `(batch, dims, ..., channels)` (channels last format, `dims=(height, width)`, `batch=1`, one image), i.e. `BHWC`.
    
    - Check ``model.input_shape`` to confirm the required dimensions of the input tensor.

* ``image`` Pillow image, corresponds to doc input.

    - Image over which to overlay the heatmap.

    - If not given, the image will be derived from ``doc`` where possible.

    - Useful if ELI5 fails in case you have a custom image model or image input.

* ``targets`` are the output classes to focus on. Possible values include: 

    - A list of integers (class ID's). *Only the first prediction from the list is currently taken*. The list must be length one. 

    - None for automatically taking the top prediction of the model.

* ``layer`` is the layer in the model from which the heatmap will be generated. Possible values are:
    
    - An instance of ``Layer``, a name (str), or an index (int)

    - None for automatically getting a suitable layer, if possible.

All other arguments are ignored.


.. note::
    Top-level :func:`eli5.explain_prediction` calls are dispatched
    to :func:`eli5.keras.explain_prediction_keras` for
    ``keras.models.Model`` and ``keras.models.Sequential``.


.. _keras-show-prediction:

show_prediction
---------------

ELI5 supports :func:`eli5.show_prediction` to conveniently 
invoke ``explain_prediction`` with ``format_as_image``, and display the explanation in an
IPython cell.


.. _keras-gradcam:

Grad-CAM
--------

ELI5 contains :func:`eli5.keras.gradcam.gradcam` and :func:`eli5.keras.gradcam.gradcam_backend`.

These functions can be used to obtain finer details of a Grad-CAM explanation.
