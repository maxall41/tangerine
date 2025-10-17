import numpy as np
from shapely import Polygon
from shapely.validation import make_valid
from skimage import measure


def outlines_list(masks: np.ndarray) -> list[Polygon]:
    """Get outlines of masks as a list of Shapely Polygons with holes.

    Uses the Slideflow approach: creates exterior polygon, then subtracts
    holes using the difference operation.
    """
    polygons = []

    for n in np.unique(masks)[1:]:  # Skip background (0)
        mn = masks == n
        if mn.sum() == 0:
            continue

        # Find all contours (external boundaries and holes)
        contours = measure.find_contours(mn, level=0.5)

        if len(contours) == 0:
            continue

        # Convert contours to (x, y) format and filter by size
        converted_contours = []
        for contour in contours:
            # find_contours returns (row, col), convert to (x, y)
            pix = np.fliplr(contour)
            if len(pix) > 4:
                converted_contours.append(pix)

        if len(converted_contours) == 0:
            continue

        # Calculate area for each contour using shoelace formula
        contour_info = []
        for contour in converted_contours:
            x = contour[:, 0]
            y = contour[:, 1]
            # Calculate signed area
            area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
            area += 0.5 * (x[-1] * y[0] - x[0] * y[-1])

            contour_info.append(
                {
                    "contour": contour,
                    "area": abs(area),
                    "signed_area": area,
                },
            )

        # Sort by absolute area (largest first)
        contour_info.sort(key=lambda x: x["area"], reverse=True)

        # The largest contour is the exterior
        exterior_contour = contour_info[0]["contour"]

        # Start with a valid exterior polygon (Slideflow approach)
        poly = make_valid(Polygon(exterior_contour))

        # Subtract holes using difference operation (Slideflow approach)
        # Process remaining contours as potential holes
        for info in contour_info[1:]:
            hole_contour = info["contour"]
            try:
                hole_poly = make_valid(Polygon(hole_contour))

                # Check if hole is contained within the current polygon
                if poly.contains(hole_poly):
                    # Subtract the hole from the polygon
                    poly = poly.difference(hole_poly)
            except Exception:
                # Skip invalid holes
                continue

        # Only add valid, non-empty polygons
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)

    return polygons
