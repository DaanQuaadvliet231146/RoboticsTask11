from stable_baselines3 import PPO
from ot2_gym_wrapper_V2 import OT2Env
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import get_custom_objects
import tensorflow as tf

# Define the custom dice loss function
def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2 * intersection + 1) / (union + 1)

# Define the custom F1 metric
def f1(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    precision = tp / (tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())
    recall = tp / (tf.reduce_sum(y_true) + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

get_custom_objects().update({"dice_loss": dice_loss, "f1": f1})

# Load the computer vision model
cv_model_path = "C:\\Users\\daanq\\Documents\\Buas Year 2\\Git\\2024-25b-fai2-adsai-DaanQuaadvliet231146\\Datalab_Tasks_Definitives\\Task_5\\Models\\DaanQuaadvliet_231146_unet_model1_256px.h5"
cv_model = load_model(cv_model_path)

# Initialise the simulation environment
num_agents = 1
env = OT2Env(render=True)
obs, info = env.reset()

image_path = env.get_plate_image()

# Load the image and process it to extract root tip positions in pixel space
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"Cannot load image from path: {image_path}")

# Preprocess the image for the CV model
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
image_resized = cv2.resize(image_gray, (256, 256))  # Resize to 256x256
image_normalized = image_resized / 255.0  # Normalize pixel values to [0, 1]
image_input = np.expand_dims(image_normalized, axis=(0, -1))  # Add batch and channel dimensions

# Predict root tip positions using the CV model
cv_predictions = cv_model.predict(image_input)
root_tip_pixels = np.argwhere(cv_predictions[0, :, :, 0] > 0.5)  # Apply thresholding

# Conversion factor: Plate dimensions in mm and pixels
plate_size_mm = 150
plate_size_pixels = 1000  # Replace with the actual size in pixels of the plate in the image
conversion_factor = plate_size_mm / plate_size_pixels

# Plate position in robot's coordinate space
plate_position_robot = np.array([0.10775, 0.088 - 0.026, 0.057])

# Convert pixel positions to robot coordinates
def pixel_to_robot_coordinates(pixel_positions, conversion_factor, plate_position_robot):
    root_tips_mm = [(x * conversion_factor, y * conversion_factor) for x, y in pixel_positions]
    root_tips_robot = [
        [x + plate_position_robot[0],
         y + plate_position_robot[1],
         plate_position_robot[2]]
        for x, y in root_tips_mm
    ]
    return root_tips_robot

# Convert to robot coordinates
goal_positions = pixel_to_robot_coordinates(root_tip_pixels, conversion_factor, plate_position_robot)

### 
# Load the Trained RL Model
#
rl_model_path = "C:\\Users\\daanq\\Documents\\BUAS_Year_2B\\Block_B_Notes\\Tasks\\Task_11\\model_baseline_jason.zip"
model = PPO.load(rl_model_path)


for goal_pos in goal_positions:
    # Set the goal position for the robot
    env.goal_position = goal_pos

    # Run the control algorithm until the robot reaches the goal position
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Calculate the distance between the pipette and the goal
        pipette_position = obs[:3]  # Current pipette position
        distance = goal_pos - pipette_position  # Goal position - pipette position

        # Calculate the error between the pipette and the goal
        error = np.linalg.norm(distance)

        # Drop the inoculum if the robot is within the required error threshold
        if error < 0.001:  # 1 mm accuracy threshold
            action = np.array([0, 0, 0, 1])  # Perform inoculation action
            obs, rewards, terminated, truncated, info = env.step(action)
            print(f"Inoculated at position: {goal_pos}, Error: {error:.4f} m")
            break

        if terminated:
            obs, info = env.reset()

print("All root tips have been inoculated.")


# Overlay the predicted root tip positions on the plate image
output_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)  # Convert grayscale back to BGR for color overlay

# Scale predicted pixel coordinates back to original image size (if needed)
height, width = image.shape[:2]
scale_x = width / 256
scale_y = height / 256

for (x, y) in root_tip_pixels:
    x = int(x * scale_x)
    y = int(y * scale_y)
    cv2.circle(output_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Draw red dot

# Save and display the annotated image
annotated_image_path = "annotated_plate_image.jpg"
cv2.imwrite(annotated_image_path, output_image)
cv2.imshow("Annotated Plate Image", output_image)
cv2.waitKey(0)  # Wait for key press to close the window
cv2.destroyAllWindows()

