import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_soil_grid(images_dict, n_per_type=1, tile_size=(240, 240), pad=12):
    """
    Display soil images in a single row (one image per soil type).
    
    Args:
        images_dict: {label: [(img_bgr, filename), ...], ...}
        n_per_type: index of image to display per type (default 1, shows first image)
        tile_size: (width, height) of each image tile
        pad: padding (kept for backward compatibility)
    """
    labels = list(images_dict.keys())
    num_types = len(labels)

    tw, th = tile_size  # width, height in pixels
    
    # Create figure with single row
    fig_w = (num_types * tw / 100) + 2
    fig_h = th / 100 + 1
    
    fig, axes = plt.subplots(1, num_types, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle('Sample Soil Images (One per Type)', fontsize=14, fontweight='bold')
    
    for col, label in enumerate(labels):
        ax = axes[0, col]
        images = images_dict.get(label, [])
        
        if len(images) > 0:
            img_bgr, filename = images[0]  # Take first image
            if img_bgr is not None and img_bgr.size > 0:
                # Convert BGR to RGB for matplotlib display
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Resize to tile_size for display
                img_display = cv2.resize(img_rgb, tile_size, interpolation=cv2.INTER_AREA)
                ax.imshow(img_display)
            else:
                ax.imshow(np.ones((th, tw, 3), dtype=np.uint8) * 235)
        else:
            # Empty tile (light gray)
            ax.imshow(np.ones((th, tw, 3), dtype=np.uint8) * 235)
        
        ax.axis('off')
        ax.set_title(label, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
