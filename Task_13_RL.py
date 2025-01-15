from stable_baselines3 import PPO
from ot2_gym_wrapper_task13 import OT2Env
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from patchify import patchify, unpatchify
from skimage.morphology import skeletonize, closing, dilation, erosion, square
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
from tensorflow.keras import backend as K
from keras.optimizers import Adam
import re
from scipy.ndimage import binary_fill_holes
from skimage.draw import line
from scipy.spatial.distance import cdist
import numpy as np
from scipy.spatial import distance
import skimage
import time



# Initialise the simulation environment
num_agents = 1
env = OT2Env(render=True)
obs, info = env.reset()

# _____________________________________________________________________________________________________________________________ #
# Step 1: Load the Computer Vision Model
model_path = "DaanQuaadvliet_231146_unet_model_it6_256px.h5"
patch_size = 256
model = load_model(model_path, compile=False)

# Function to crop the plate image
def crop_image(image):
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image  # Return original if no contours found

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image, x, y  # Return cropped image and offsets

# Crop the plate image
image_path = env.get_plate_image()
plate_image = cv2.imread(image_path, 0)  # Load the plate image in grayscale

# Crop the plate image
cropped_image, x_offset, y_offset = crop_image(plate_image)


# _____________________________________________________________________________________________________________________________ #
# Predict the mask
def predict_mask(image, model, patch_size):
    normalized_image = image / 255.0
    desired_height = (image.shape[0] // patch_size + 1) * patch_size
    desired_width = (image.shape[1] // patch_size + 1) * patch_size
    padded_image = cv2.copyMakeBorder(
        normalized_image, 0, desired_height - image.shape[0], 0, desired_width - image.shape[1],
        cv2.BORDER_CONSTANT, value=0
    )
    patches = patchify(padded_image, (patch_size, patch_size), step=patch_size)
    patches = patches.reshape(-1, patch_size, patch_size, 1)
    predicted_patches = model.predict(patches)
    predicted_patches = (predicted_patches > 0.5).astype(np.uint8)
    patch_shape = (padded_image.shape[0] // patch_size, padded_image.shape[1] // patch_size)
    predicted_patches = predicted_patches.reshape(patch_shape[0], patch_shape[1], patch_size, patch_size)
    predicted_mask = unpatchify(predicted_patches, padded_image.shape)
    return predicted_mask[:image.shape[0], :image.shape[1]]  # Crop back to original dimensions

# Predict the mask for the cropped image
predicted_mask = predict_mask(cropped_image, model, patch_size)

# Preprocess and skeletonize the mask
def preprocess_and_skeletonize(mask):
    dilated = dilation(mask, square(10))
    closed = closing(dilated, square(9))
    filled = binary_fill_holes(closed)
    eroded = erosion(filled, square(4))
    skeleton = skeletonize(eroded)
    return skeleton

skeletonized_mask = preprocess_and_skeletonize(predicted_mask)

# Crop skeletonized mask (if needed)
skeletonized_mask = skeletonized_mask[300:, 10:-10]

# _____________________________________________________________________________________________________________________________ #
# Detect root tips
def detect_root_tips(skeletonized_mask):
    labeled_mask = label(skeletonized_mask, connectivity=2)
    root_tips = []
    for region in regionprops(labeled_mask):
        if region.area >= 40:  # Minimum area threshold
            coords = region.coords
            bottom_point = coords[np.argmax(coords[:, 0])]  # Bottom-most point
            root_tips.append(bottom_point)
    return root_tips

root_tips = detect_root_tips(skeletonized_mask)

# Convert pixel coordinates to robot coordinates
plate_size_m = 0.15  # Plate size in mm
plate_position_robot = [0.10775, 0.062, 0.057]  # Plate's top-left corner in robot coordinates

def convert_to_robot_coordinates(root_tips, x_offset, y_offset, image_width_pixels, plate_position_robot, plate_size_m):
    conversion_factor_m = plate_size_m / image_width_pixels
    robot_coords = []
    for tip in root_tips:
        m_x = (tip[1] + x_offset) * conversion_factor_m
        m_y = (tip[0] + y_offset) * conversion_factor_m
        robot_x = m_x + plate_position_robot[0]
        robot_y = m_y + plate_position_robot[1]
        robot_z = plate_position_robot[2]

        robot_coords.append([robot_x, robot_y, robot_z])
        print("Pixel-based root tip: ", tip)
        print("Converted to robot coords (m): ", [robot_x, robot_y, robot_z])
    return robot_coords

robot_coordinates = convert_to_robot_coordinates(
    root_tips, x_offset, y_offset, skeletonized_mask.shape[1], plate_position_robot, plate_size_m
)
# _____________________________________________________________________________________________________________________________ #
# Load the trained RL model
rl_model = PPO.load("models\\Final_model_it2")

for goal_pos in robot_coordinates:
    # Set the goal position for the robot
    env.goal_position = goal_pos
    # Run the control algorithm until the robot reaches the goal position
    while True:
        action, _states = rl_model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info  = env.step(action)
        # calculate the distance between the pipette and the goal
        distance = obs[3:] - obs[:3] # goal position - pipette positiond0
        # calculate the error between the pipette and the goal
        error = np.linalg.norm(distance)
        # Drop the inoculum if the robot is within the required error
        if error < 0.01: # 10mm is used as an example here it is too large for the real use case
            action = np.array([0, 0, 0, 1])
            obs, rewards, terminated, truncated, info  = env.step(action)
            break

        if terminated:
            obs, info = env.reset()
# _____________________________________________________________________________________________________________________________ #
# _____________________________________________________________________________________________________________________________ #
# _____________________________________________________________________________________________________________________________ #
# _____________________________________________________________________________________________________________________________ #

