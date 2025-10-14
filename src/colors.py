import copy
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.color import hed2rgb, rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


def extract_heamtoxylin(input: np.ndarray):
    hed = rgb2hed(input)
    h_channel = hed[:, :, 0]
    hed_hematoxylin_only = np.zeros_like(hed)
    hed_hematoxylin_only[:, :, 0] = h_channel
    hematoxylin_rgb = hed2rgb(hed_hematoxylin_only)

    p_low, p_high = np.percentile(h_channel, [1, 99])
    h_channel_clipped = np.clip(h_channel, p_low, p_high)

    heamtoxylin_threshold = threshold_otsu(h_channel_clipped)
    heamtoxylin_mask = h_channel > heamtoxylin_threshold

    return hematoxylin_rgb, heamtoxylin_mask


def remove_white_background(image, threshold=240, min_brightness=200):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Extract RGB channels
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Create mask for white/near-white pixels using multiple criteria:
    # 1. All RGB channels are above threshold (pure white detection)
    mask_pure_white = (r >= threshold) & (g >= threshold) & (b >= threshold)

    # 2. Average brightness is high (catches light grays and off-whites)
    avg_brightness = (r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32)) / 3
    mask_bright = avg_brightness >= min_brightness

    # 3. Low color variation (standard deviation across RGB channels is small)
    # This catches uniform light pixels
    rgb_stack = np.stack([r, g, b], axis=-1).astype(np.float32)
    color_std = np.std(rgb_stack, axis=-1)
    mask_uniform = color_std < 15  # Low variation means nearly uniform color

    # Combine masks: remove pixels that are bright AND uniform
    background_mask = mask_bright & mask_uniform

    # Also always remove pure white pixels
    background_mask = background_mask | mask_pure_white

    # Create RGB image with black background
    result = image.copy()
    result[background_mask] = 0  # Set background to black

    return result


def remove_white_background_adaptive(image, percentile=95):
    # Calculate average brightness for each pixel
    avg_brightness = np.mean(image, axis=2)

    print("avg_brightness", avg_brightness)

    # Determine threshold based on percentile
    threshold_start = np.percentile(avg_brightness, percentile)
    threshold = int(threshold_start * 0.95)
    min_brightness = int(threshold_start * 0.85)

    print("threshold", threshold)

    # Use the main function with calculated threshold
    return (
        remove_white_background(
            image,
            threshold=threshold,  # Slightly lower to catch near-threshold pixels
            min_brightness=min_brightness,
        ),
        threshold,
        min_brightness,
    )


from scipy.ndimage import gaussian_filter


def identify_blobs(image: np.ndarray) -> np.ndarray:
    distance = ndi.distance_transform_edt(image)

    # Smooth distance map to merge nearby peaks
    distance_smooth = gaussian_filter(distance, sigma=2.0)  # Adjust sigma

    coords = peak_local_max(
        distance_smooth,
        footprint=np.ones((3, 3)),
        labels=image,
        min_distance=1,
    )

    watershed_mask = np.zeros(distance.shape, dtype=bool)
    watershed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(watershed_mask)
    labels = watershed(-distance, markers, mask=image, watershed_line=True)
    return labels


@dataclass
class OilRedOQuantificationResults:
    # Standard
    oil_red_o_area: float
    total_epithelial_area: float
    oil_red_o_percent_coverage: float
    # Counts
    oil_red_o_droplet_count: int
    oil_red_o_droplet_mean_area: float
    # Image
    oil_red_o_cutout_image: np.ndarray


# H OD Vectors: (0.651, 0.701, 0.29)
def quantify_oil_red_o_stain(
    cropped_image: np.ndarray,
    uncropped_image: None | np.ndarray = None,
) -> OilRedOQuantificationResults:
    if uncropped_image is None:
        uncropped_image = cropped_image

    # im = Image.fromarray(cropped_image)
    # im.save("raw_image.tif")

    heamtoxylin, heamtoxylin_mask = extract_heamtoxylin(uncropped_image)
    original_image = copy.deepcopy(cropped_image)
    cropped_image[heamtoxylin_mask] = [0, 0, 0]

    # im = Image.fromarray((heamtoxylin * 255).astype(np.uint8))
    # im.save("hematoxylin_rgb.tif")

    # im = Image.fromarray(cropped_image)
    # im.save("oilred_o_pre_white.tif")

    epithelial_bkg, threshold, min_brightness = remove_white_background_adaptive(original_image)  # threshold 141.0
    cropped_image = remove_white_background(cropped_image, threshold, min_brightness)

    # im = Image.fromarray(cropped_image)
    # im.save("oilred_o.tif")

    # im = Image.fromarray(heamtoxylin_mask)
    # im.save("heamtoxylin_mask.tif")

    mask = np.dot(cropped_image[..., :3], [0.2989, 0.5870, 0.1140])
    mask = mask > 1

    oil_red_o_area = np.sum(mask)
    total_area = np.sum(epithelial_bkg > 0)

    oil_red_o_percent_coverage = (oil_red_o_area / total_area) * 100

    print("Finished Oil Red O Extraction")

    blobs = identify_blobs(mask)

    print("Finished Watershed segmentation")

    unique_labels, blob_sizes = np.unique(blobs, return_counts=True)

    # Remove background (label 0)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
        blob_sizes = blob_sizes[1:]

    num_blobs = len(unique_labels)
    oil_red_o_droplet_mean_area = float(np.mean(blob_sizes) if len(blob_sizes) > 0 else 0)

    print("Finished counting segmented droplets")

    print("oil_red_o_area", oil_red_o_area)
    print("total_area", total_area)
    print("oil_red_o_percent_coverage", oil_red_o_percent_coverage)
    print("num of oil red o droplets: ", num_blobs)
    print("mean size of oil red o droplets: ", oil_red_o_droplet_mean_area)

    # im = Image.fromarray(cropped_image)
    # im.save("oilred_o_mask.tif")

    # im = Image.fromarray(mask)
    # im.save("mask.tif")

    # blobs_colored = label2rgb(blobs, bg_label=0)
    # blobs_colored = (blobs_colored * 255).astype(np.uint8)
    # im = Image.fromarray(blobs_colored)
    # im.save("blobs.tif")

    # im = Image.fromarray(epithelial_bkg)
    # im.save("epithelial_bkg.tif")

    return OilRedOQuantificationResults(
        oil_red_o_cutout_image=cropped_image,
        oil_red_o_area=oil_red_o_area,
        total_epithelial_area=total_area,
        oil_red_o_percent_coverage=oil_red_o_percent_coverage,
        oil_red_o_droplet_count=num_blobs,
        oil_red_o_droplet_mean_area=oil_red_o_droplet_mean_area,
    )
