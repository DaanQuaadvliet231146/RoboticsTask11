from stable_baselines3 import PPO
from ot2_gym_wrapper_task13 import OT2Env
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from skimage.morphology import skeletonize, closing, dilation, erosion, square
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from patchify import unpatchify, patchify
import time
import matplotlib.pyplot as plt
from simple_pid import PID




# Constants
PLATE_SIZE_MM = 150  # Plate dimensions in mm
PLATE_POSITION_ROBOT = [0.10775, 0.062, 0.057]  # Top-left corner of the plate in robot space



# Helper Functions
def visualize_detection(image, root_tips, title="Root Tips Detected"):
    """Visualize root tip detection."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    for tip in root_tips:
        plt.scatter(tip[1], tip[0], color="red", s=100, label=f"Root Tip: {tip}")
    plt.title(title)
    plt.legend()
    plt.show()

def crop_image(image):
    # Apply binary threshold to detect the largest contour
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found. Returning original image.")
        return image  # Return original if no contours found

    # Get the largest contour and crop
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image, (x, y, w, h)

# Helper Functions
def preprocess_and_skeletonize(mask):
    """Preprocess and skeletonize the mask."""
    dilated = dilation(mask, square(10))
    closed = closing(dilated, square(8))
    filled = binary_fill_holes(closed)
    eroded = erosion(filled, square(4))
    skeleton = skeletonize(eroded)
    return skeleton

def predict_mask(image, model, patch_size=256):
    """Predict the mask using the U-Net model."""
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
    return predicted_mask[:image.shape[0], :image.shape[1]]

def detect_root_tips(skeletonized_mask, num_regions=5, min_area_threshold=40):
    """
    Detect root tips in the skeletonized mask by dividing it into regions.
    
    Args:
        skeletonized_mask (np.ndarray): Binary skeletonized mask.
        num_regions (int): Number of vertical regions to divide the image into.
        min_area_threshold (int): Minimum area for a region to be considered.
    
    Returns:
        list of tuple: List of root tip coordinates (row, col) for each region.
    """
    height, width = skeletonized_mask.shape
    region_width = width // num_regions  # Width of each region

    root_tips = []  # Store the root tips for each region

    for region_idx in range(num_regions):
        # Define the boundaries of the current region
        x_start = region_idx * region_width
        x_end = (region_idx + 1) * region_width if region_idx < num_regions - 1 else width

        # Extract the region
        region_mask = skeletonized_mask[:, x_start:x_end]

        # Label the regions and find root tips
        labeled_mask = label(region_mask, connectivity=2)
        root_tip = None
        max_y = -1  # Start with an invalid value for maximum y-coordinate

        for region in regionprops(labeled_mask):
            if region.area >= min_area_threshold:
                coords = region.coords
                bottom_point = coords[np.argmax(coords[:, 0])]  # Bottom-most point

                # Update if the current bottom point is lower
                if bottom_point[0] > max_y:
                    max_y = bottom_point[0]
                    root_tip = (bottom_point[0], bottom_point[1] + x_start)  # Adjust x-coordinate to global

        if root_tip is not None:
            root_tips.append(root_tip)

    return root_tips

def calculate_conversion_factors(plate_width_pixels, plate_height_pixels):
    conversion_factor_x = PLATE_SIZE_MM / plate_width_pixels
    conversion_factor_y = PLATE_SIZE_MM / plate_height_pixels
    return conversion_factor_x, conversion_factor_y

def convert_pixel_to_robot_coordinates(root_tips_px, conversion_factors, plate_position_robot):
    """
    Convert pixel coordinates (relative to cropped image) to robot coordinates.

    Args:
        root_tips_px (list of tuple): List of root tip pixel coordinates (row, col).
        conversion_factors (tuple): Conversion factors for x and y (pixels to mm).
        plate_position_robot (list): Position of the plate's top-left corner in robot space.

    Returns:
        list of list: List of robot coordinates [x, y, z].
    """
    conversion_factor_x, conversion_factor_y = conversion_factors

    robot_coordinates = []
    for (row, col) in root_tips_px:
        # Convert directly from cropped image coordinates to mm
        tip_x_mm = row * conversion_factor_x
        tip_y_mm = col * conversion_factor_y

        # Convert to robot coordinates
        robot_x = plate_position_robot[0] + tip_x_mm / 1000  # Convert mm to meters
        robot_y = plate_position_robot[1] + tip_y_mm / 1000  # Convert mm to meters
        robot_z = plate_position_robot[2] + 0.12  # Adjust z to avoid collision

        robot_coordinates.append([robot_x, robot_y, robot_z])

        # Debugging Outputs
        print(f"Root Tip Pixel: ({row}, {col})")
        print(f"Root Tip MM: ({tip_x_mm}, {tip_y_mm})")
        print(f"Root Tip Robot: ({robot_x}, {robot_y}, {robot_z})")

    return robot_coordinates

class PIDControllerPipeline:
    def __init__(self, env=None):
        # Reuse the existing environment or create a new one without rendering
        self.env = env if env else OT2Env(render=False)
        self.dt = 0.9  # PID time step
        self.pid_x = PID(5.0, 0.01, 0.05, sample_time=self.dt)
        self.pid_y = PID(5.0, 0.01, 0.05, sample_time=self.dt)
        self.pid_z = PID(5.0, 0.01, 0.05, sample_time=self.dt)

        for pid in [self.pid_x, self.pid_y, self.pid_z]:
            pid.output_limits = (-4.0, 4.0)

    def move_to_position(self, target_position):
        observation, _ = self.env.reset()
        pipette_position = observation[:3]

        self.pid_x.setpoint = target_position[0]
        self.pid_y.setpoint = target_position[1]
        self.pid_z.setpoint = target_position[2]

        for _ in range(1000):  # Max steps
            vel_x = self.pid_x(pipette_position[0])
            vel_y = self.pid_y(pipette_position[1])
            vel_z = self.pid_z(pipette_position[2])

            action = [vel_x, vel_y, vel_z]
            observation, _, _, _, _ = self.env.step(action)
            pipette_position = observation[:3]

            if np.linalg.norm(pipette_position - target_position) < 0.001:
                print(f"Reached target: {target_position}")
                return True

        print(f"Failed to reach target: {target_position}")
        return False


# Main Code
if __name__ == "__main__":
    # Initialize the environment
    env = OT2Env(render=True)
    obs, info = env.reset()

    # Load the U-Net model
    model_path = "DaanQuaadvliet_231146_unet_model_it6_256px.h5"
    unet_model = load_model(model_path, compile=False)

    # Get plate image
    image_path = env.get_plate_image()
    plate_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Use the crop_image function
    cropped_plate_image, (x_offset, y_offset, plate_width_px, plate_height_px) = crop_image(plate_image)

    # Predict mask and detect root tips
    predicted_mask = predict_mask(cropped_plate_image, unet_model, patch_size=256)
    skeletonized_mask = preprocess_and_skeletonize(predicted_mask)
    root_tips_px = detect_root_tips(skeletonized_mask)

        # Convert pixel to robot coordinates
    conversion_factors = calculate_conversion_factors(plate_width_px, plate_height_px)
    robot_coordinates = convert_pixel_to_robot_coordinates(
        root_tips_px, conversion_factors, PLATE_POSITION_ROBOT
    )

    print(f"Robot Coordinates: {robot_coordinates}")

    pid_controller = PIDControllerPipeline()
    for idx, goal in enumerate(robot_coordinates, start=1):
        print(f"Moving to root tip {idx}: {goal}")
        pid_controller.move_to_position(goal)