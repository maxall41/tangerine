"""A large part of the code in this file was taken from the Slideflow project released under the Apache 2.0 license."""

import json
from collections.abc import Callable
from os.path import dirname, exists, isdir, join
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def topleft_pad_torch(img, size, padval=0):
    """Pad an image to the top-left.

    Args:
        img (torch.Tensor): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    # Pad to target size.
    if img.shape[0] < size:
        pad_x = size - img.shape[0]
    else:
        pad_x = 0
    if img.shape[1] < size:
        pad_y = size - img.shape[1]
    else:
        pad_y = 0

    # PyTorch requires padding in the form (pad_channel_start, pad_channel_end, pad_top, pad_bottom, pad_left, pad_right).
    padded = F.pad(img, (0, 0, 0, pad_y, 0, pad_x), mode="constant", value=padval)

    return padded


def topleft_pad_numpy(img, size, padval=0):
    """Pad an image to the top-left.

    Args:
        img (np.ndarray): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    # Pad to target size.
    if img.shape[0] < size:
        pad_x = size - img.shape[0]
    else:
        pad_x = 0
    if img.shape[1] < size:
        pad_y = size - img.shape[1]
    else:
        pad_y = 0
    padded = np.pad(img, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant", constant_values=padval)
    return padded


def topleft_pad(img, size, padval=0):
    """Pad an image to the top-left.

    Args:
        img (np.ndarray or torch.Tensor): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    if isinstance(img, np.ndarray):
        return topleft_pad_numpy(img, size, padval)
    if isinstance(img, torch.Tensor):
        return topleft_pad_torch(img, size, padval)
    raise ValueError(f"Unknown image type: {type(img)}")


def make_tiles(
    imgi: np.ndarray,
    bsize: int = 224,
    augment: bool = False,
    tile_overlap: float = 0.1,
) -> tuple[np.ndarray, list, list, int, int]:
    """Make tiles from an image.

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles


    """
    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[:, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]]

    return IMG, ysub, xsub, Ly, Lx


def _taper_mask(ly: int = 224, lx: int = 224, sig: float = 7.5) -> np.ndarray:
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1 / (1 + np.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[
        bsize // 2 - ly // 2 : bsize // 2 + ly // 2 + ly % 2,
        bsize // 2 - lx // 2 : bsize // 2 + lx // 2 + lx % 2,
    ]
    return mask


def average_tiles(
    y: np.ndarray,
    ysub: list,
    xsub: list,
    Ly: int,
    Lx: int,
) -> np.ndarray:
    """Average results of network over tiles.

    Parameters
    ----------
    y: float, [ntiles x nclasses x bsize x bsize]
        output of cellpose network for each tile

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    Ly : int
        size of pre-tiled image in Y (may be larger than original image if
        image size is less than bsize)

    Lx : int
        size of pre-tiled image in X (may be larger than original image if
        image size is less than bsize)

    Returns
    -------
    yf: float32, [nclasses x Ly x Lx]
        network output averaged over tiles

    """
    Navg = np.zeros((Ly, Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0] : ysub[j][1], xsub[j][0] : xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0] : ysub[j][1], xsub[j][0] : xsub[j][1]] += mask
    yf /= Navg
    return yf


class SegmentModel(pl.LightningModule):
    losses = {
        "dice": smp.losses.DiceLoss,
        "jaccard": smp.losses.JaccardLoss,
        "focal": smp.losses.FocalLoss,
        "tversky": smp.losses.TverskyLoss,
        "lovasz": smp.losses.LovaszLoss,
        "bce": smp.losses.SoftBCEWithLogitsLoss,
        "ce": smp.losses.SoftCrossEntropyLoss,
        "mcc": smp.losses.MCCLoss,
    }

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        out_classes: int,
        *,
        mpp: Optional[float] = None,
        lr: float = 1e-4,
        loss: str | Callable = "dice",
        mode: str = "binary",
        **kwargs,
    ):
        super().__init__()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.mpp = mpp
        self.lr = lr
        self.out_classes = out_classes

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.mode = mode
        self.loss_fn = self.get_loss_fn(loss, mode)
        self.outputs = {
            "train": [],
            "valid": [],
        }

    @staticmethod
    def get_loss_fn(loss: str | Callable, mode: str) -> Callable:
        if not isinstance(loss, str):
            return loss
        if loss in SegmentModel.losses:
            loss_fn = SegmentModel.losses[loss]
        else:
            raise ValueError(f"Invalid loss: {loss}")

        if loss in ("bce", "ce", "mcc"):
            if mode and mode != "binary":
                raise ValueError(f"Invalid loss mode for loss {loss!r}: Expected 'binary', got: {mode!r}")
            return loss_fn()
        return loss_fn(mode=SegmentModel.get_loss_mode(mode))

    @staticmethod
    def get_loss_mode(mode):
        if mode == "binary":
            return smp.losses.BINARY_MODE
        if mode == "multiclass":
            return smp.losses.MULTICLASS_MODE
        if mode == "multilabel":
            return smp.losses.MULTILABEL_MODE
        raise ValueError(f"Invalid loss mode: {mode}")

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        if self.mode == "binary":
            # Shape of the mask should be [batch_size, num_classes, height, width]
            # for binary segmentation num_classes = 1
            assert mask.ndim == 4
        elif self.mode == "multiclass":
            # Shape of the mask should be [batch_size, height, width]
            # for multiclass segmentation, values are the classes.
            assert mask.ndim == 3
        elif self.mode == "multilabel":
            # Shape of the mask should be [batch_size, num_classes, height, width]
            assert mask.ndim == 4

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        if self.mode == "multiclass":
            prob_mask = torch.softmax(logits_mask, dim=1)
            pred_mask = torch.argmax(prob_mask, dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(),
                mask.long(),
                mode=self.mode,
                num_classes=self.out_classes,
            )
        else:
            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(),
                mask.long(),
                mode=self.mode,
            )

        output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.outputs[stage].append(output)
        return output

    def shared_epoch_end(self, stage):
        outputs = self.outputs[stage]

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou.to(self.device).float(),
            f"{stage}_dataset_iou": dataset_iou.to(self.device).float(),
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.outputs[stage].clear()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def run_tiled_inference(self, img: np.ndarray):
        """Run inference on an image, with tiling."""
        # Pad to at least the target size.
        if img.shape[-1] == 4:
            img = img[..., :3]
        orig_dims = img.shape
        img = topleft_pad(img, 1024).transpose(2, 0, 1)

        # Tile the thumbnail.
        tiles, ysub, xsub, ly, lx = make_tiles(img, 1024)
        batched_tiles = tiles.reshape(-1, 3, 1024, 1024)

        # Generate UNet predictions.
        with torch.inference_mode():
            tile_preds = []
            for tile in batched_tiles:
                pred = self.forward(torch.from_numpy(tile).unsqueeze(0).to(self.device))
                tile_preds.append(pred)
            tile_preds = torch.cat(tile_preds)

        # Merge predictions across the tiles.
        tiled_preds = average_tiles(tile_preds.cpu().numpy(), ysub, xsub, ly, lx)

        # Crop predictions to the original size.
        tiled_preds = tiled_preds[: orig_dims[0], : orig_dims[1]]

        # Softmax, if multiclass.
        if self.mode == "binary":
            tiled_preds = tiled_preds[0]
        elif self.mode == "multiclass":
            tiled_preds = torch.from_numpy(tiled_preds).softmax(dim=0).numpy()

        return tiled_preds


class SegmentConfig:
    def __init__(
        self,
        arch: str = "FPN",
        encoder_name: str = "resnet34",
        *,
        size: int = 1024,
        in_channels: int = 3,
        out_classes: Optional[int] = None,
        train_batch_size: int = 8,
        val_batch_size: int = 16,
        epochs: int = 8,
        mpp: float = 20,
        lr: float = 1e-4,
        loss: str = "dice",
        mode: str = "binary",
        labels: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Configuration for a segmentation model.

        Args:
            arch (str): Model architecture. Defaults to 'FPN'.
            encoder_name (str): Encoder name. Defaults to 'resnet34'.

        Keyword Args:
            size (int): Size of input images. Defaults to 1024.
            in_channels (int): Number of input channels. Defaults to 3.
            out_classes (int, optional): Number of output classes.
                If None, will attempt to auto-detect based the provided labels
                and loss mode. If labels are not provided, it defaults to 1.
            train_batch_size (int): Training batch size. Defaults to 8.
            val_batch_size (int): Validation batch size. Defaults to 16.
            epochs (int): Number of epochs to train for. Defaults to 8.
            mpp (float): MPP to use for training. Defaults to 10.
            loss (str): Loss function. Defaults to 'dice'.
            mode (str): Loss mode. Can be 'binary', 'multiclass', or 'multilabel'.
                Defaults to 'binary'.
            labels (List[str]): Names for ROI labels. Only used if mode
                is 'multiclass'. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model.

        """
        self.arch = arch
        self.encoder_name = encoder_name
        self.size = size
        self.in_channels = in_channels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.mpp = mpp
        self.lr = lr
        self.loss = loss
        self.mode = mode
        self.labels = labels
        self.kwargs = kwargs
        if out_classes is None:
            if mode == "binary" or labels is None:
                self.out_classes = 1
            elif mode == "multiclass":
                self.out_classes = len(labels) + 1
            elif mode == "multilabel":
                self.out_classes = len(labels)
        else:
            self.out_classes = out_classes

    def __repr__(self) -> str:
        return (
            f"SegmentConfig(\n"
            f"    arch={self.arch!r},\n"
            f"    encoder_name={self.encoder_name!r},\n"
            f"    size={self.size!r},\n"
            f"    in_channels={self.in_channels!r},\n"
            f"    out_classes={self.out_classes!r},\n"
            f"    train_batch_size={self.train_batch_size!r},\n"
            f"    val_batch_size={self.val_batch_size!r},\n"
            f"    epochs={self.epochs!r},\n"
            f"    mpp={self.mpp!r},\n"
            f"    lr={self.lr!r},\n"
            f"    loss={self.loss!r},\n"
            f"    mode={self.mode!r},\n"
            f"    labels={self.labels!r},\n"
            f"    **{self.kwargs!r}\n"
            f")"
        )

    @classmethod
    def from_json(cls, path: str) -> "SegmentConfig":
        """Load a configuration from a JSON file.

        Args:
            path (str): Path to JSON file.

        Returns:
            SegmentConfig: SegmentConfig object.

        """
        with open(path) as data_file:
            data = json.load(data_file)
        params = data["params"].copy()
        del data["params"]
        return cls(**data, **params)

    def build_model(self) -> SegmentModel:
        """Build a segmentation model from this configuration."""
        return SegmentModel(
            self.arch,
            self.encoder_name,
            in_channels=self.in_channels,
            out_classes=self.out_classes,
            mpp=self.mpp,
            loss=self.loss,
            mode=self.mode,
            **self.kwargs,
        )


def load_model_and_config(path: str):
    """Load a model and its configuration from a path.

    Args:
        path (str): Path to model file, or directory containing model file.

    Returns:
        Tuple[SegmentModel, SegmentConfig]: Tuple of model and configuration.

    """
    if not exists(path):
        raise ValueError(f"Model '{path}' does not exist.")
    if isdir(path):
        path = join(path, "model.pth")
    if not path.endswith("pth"):
        raise ValueError(f"Model '{path}' is not a valid model path.")

    # Load the model configuration.
    cfg_path = join(dirname(path), "segment_params.json")
    if not exists(cfg_path):
        raise ValueError(f"Model '{path}' does not contain a segment_params.json file.")
    cfg = SegmentConfig.from_json(cfg_path)

    # Build the model.
    model = cfg.build_model()

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    # Load the weights.
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    return model, cfg
