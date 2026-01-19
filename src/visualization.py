import cv2
import numpy as np

def show_soil_grid(images_dict, n_per_type=5, tile_size=(240, 240), pad=12):
    """
    Open a single OpenCV window showing one row per soil type and the first N images.
    images_dict: {label: [(img_bgr, filename), ...], ...}
    """
    labels = list(images_dict.keys())
    rows = len(labels)
    cols = n_per_type

    tw, th = tile_size  # width, height
    header_h = 30       # height reserved for soil label

    canvas_w = pad + cols * (tw + pad)
    canvas_h = pad + rows * (th + header_h + pad)

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for r, label in enumerate(labels):
        images = images_dict.get(label, [])

        y_header = pad + r * (th + header_h + pad)
        y0 = y_header + header_h

        # soil type centered above the row
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        text_x = (canvas_w - text_size[0]) // 2
        text_y = y_header + 22

        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        for c in range(cols):
            x0 = pad + c * (tw + pad)
            tile = np.full((th, tw, 3), 235, dtype=np.uint8)

            if c < len(images):
                img_bgr, _ = images[c]
                if img_bgr is not None and img_bgr.size > 0:
                    tile = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)

            canvas[y0:y0 + th, x0:x0 + tw] = tile

    cv2.namedWindow("Soil visualization", cv2.WINDOW_NORMAL)
    cv2.imshow("Soil visualization", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
