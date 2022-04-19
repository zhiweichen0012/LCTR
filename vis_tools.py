import numpy as np
import torch
from numpy.lib.function_base import blackman
from skimage import measure

import os
import cv2
import random
import colorsys
import pdb
from skimage import measure
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def cam_gen_box(scoremap_image, threshold):
    scoremap_image = scoremap_image * 255.0
    _CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
    scoremap_image = np.expand_dims((scoremap_image).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY,
    )
    contours = cv2.findContours(
        image=thr_gray_heatmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[_CONTOUR_INDEX]

    if len(contours) == 0:
        return [0, 0, 224, 224]

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, w + x, h + y]


def resize_cam(cam, size=(224, 224), mask=False):
    cam = cv2.resize(cam, (size[0], size[1]))
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam


def return_box_cam(attn_avg, attn_conv, th=0.1):
    attn_avg = attn_avg.detach().cpu().numpy()
    if attn_conv != None:
        attn_conv = attn_conv.detach().cpu().numpy()
        map = attn_avg * attn_conv
    else:
        map = attn_avg
    cam_b = resize_cam(map)
    box = cam_gen_box(cam_b, th)
    return box, cam_b


def norm_atten_map(attn_map):
    min_val = np.min(attn_map)
    max_val = np.max(attn_map)
    attn_norm = (attn_map - min_val) / (max_val - min_val + 1e-15)
    if max_val - min_val == 0:
        return np.zeros_like(attn_map)
    return attn_norm


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = (
            image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        )
    return image


def display_instances(
    image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5
):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print("{name} saved.")
    return


def re_pca_box(heatmap, size=224):
    boxes = []
    if heatmap.ndim == 3:
        for he in heatmap:
            highlight = np.zeros(he.shape)
            highlight[he > 0] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1

            # visualize heatmap
            # show highlight in origin image
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(
                highlight, (size, size), interpolation=cv2.INTER_NEAREST
            )
            props = measure.regionprops(highlight_big.astype(int))
            if len(props) == 0:
                bbox = [0, 0, size, size]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]
            temp_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            boxes.append(temp_bbox)
        return boxes  # x,y,w,h
    else:
        highlight = np.zeros(heatmap.shape)
        highlight[heatmap > 0] = 1
        # max component
        all_labels = measure.label(highlight)
        highlight = np.zeros(highlight.shape)
        highlight[all_labels == count_max(all_labels.tolist())] = 1

        # visualize heatmap
        # show highlight in origin image
        highlight = np.round(highlight * 255)
        highlight_big = cv2.resize(
            highlight, (size, size), interpolation=cv2.INTER_NEAREST
        )
        props = measure.regionprops(highlight_big.astype(int))
        if len(props) == 0:
            bbox = [0, 0, size, size]
        else:
            temp = props[0]['bbox']
            bbox = [temp[1], temp[0], temp[3], temp[2]]
        return bbox


def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item == 0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d: d[1], reverse=True)
    return count_dict[0][0]

