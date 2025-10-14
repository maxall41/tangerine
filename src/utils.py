import os

import numpy as np
import torch
from PyQt6.QtGui import QImage
from slideflow import segment


def to_qimage(arr):
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
            return qimg.copy()
        a = arr.astype(np.float64)
        mn, mx = a.min(), a.max()
        if mx > mn:
            a = (255.0 * (a - mn) / (mx - mn)).astype(np.uint8)
        else:
            a = np.zeros_like(a, dtype=np.uint8)
        h, w = a.shape
        qimg = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8)
        return qimg.copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            mn, mx = a.min(), a.max()
            if mx > mn:
                a = (255.0 * (a - mn) / (mx - mn)).astype(np.uint8)
            else:
                a = np.zeros_like(a, dtype=np.uint8)
        else:
            a = arr
        h, w, _ = a.shape
        bytes_per_line = 3 * w
        qimg = QImage(a.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg.copy()
    raise ValueError("Unsupported array shape for display")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def auto_segment(path_to_image):
    model, config = segment.load_model_and_config(os.path.join(os.getcwd(), "./model/segment.pth"))
    if torch.backends.mps.is_available():
        model = model.to("mps")
    print("model device", model.device)
    pred = model.run_slide_inference(path_to_image)
    print("Finished auto segmentation!")
    return sigmoid(pred)


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
