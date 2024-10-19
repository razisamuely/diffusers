import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import os
import shutil
import math

def estimate_face_angle(landmarks, image_width, image_height):
    # Get the centers of both eyes
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Calculate the vector from the left eye to the right eye
    eye_vector = [right_eye.x - left_eye.x, right_eye.y - left_eye.y]

    # Calculate the angle between the eye vector and the horizontal axis
    angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))

    return abs(angle)

def crop_face_mesh(image_path, max_angle=30):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Estimate face angle
        face_angle = estimate_face_angle(landmarks, image.shape[1], image.shape[0])
        
        # Create a mask for the face mesh
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Get image dimensions
        ih, iw = image.shape[:2]
        
        # Draw face mesh on the mask
        face_points = [(int(l.x * iw), int(l.y * ih)) for l in landmarks]
        hull = cv2.convexHull(np.array(face_points))
        cv2.fillConvexPoly(mask, hull, 255)

        # Apply the mask to the image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Crop to the bounding box of the face mesh
        x, y, w, h = cv2.boundingRect(hull)
        cropped = result[y:y+h, x:x+w]

        return cropped, face_angle
    else:
        print("No face detected in the image.")
        return None, None

# Example usage
if __name__ == "__main__":
    folder_path = "/home/raz/workspace/diffusers/examples/controlnet/prepare_data/images"
    # delete the images_cropped and high_angle_faces folders if they exist
    if os.path.exists("images_cropped"):
        shutil.rmtree("images_cropped")
    if os.path.exists("high_angle_faces"):
        shutil.rmtree("high_angle_faces")
    
    # create folders
    os.makedirs("images_cropped", exist_ok=True)
    os.makedirs("high_angle_faces", exist_ok=True)

    max_angle = 1.2  # You can adjust this value

    for sub_folder in Path(folder_path).iterdir():
        if sub_folder.is_dir():
            images = [x for x in sub_folder.iterdir() if x.is_file() and x.suffix in [".jpg", ".jpeg", ".png"]]
            for image in images:
                image_name = image.stem
                print(f"Processing {image_name}...")
                cropped_face, face_angle = crop_face_mesh(str(image), max_angle)
                
                if cropped_face is not None:
                    if face_angle <= max_angle:
                        target_dir = Path(f"images_cropped/{sub_folder.name}")
                        target_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(target_dir / f"{image_name}.png"), cropped_face)
                        print(f"Saved cropped image to {target_dir / f'{image_name}.png , face angle: {face_angle:.2f} degrees'}")
                    else:
                        target_dir = Path(f"high_angle_faces/{sub_folder.name}")
                        target_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(target_dir / f"{image_name}_angle_{face_angle:.2f}.png"), cropped_face)
                        print(f"Face angle too high: {face_angle:.2f} degrees. Saved to {target_dir / f'{image_name}_angle_{face_angle:.2f}.png'}")

                # print number of images in parent folder with os.listdir
                print(f"Number of images in parent folder: {len(os.listdir(folder_path))}")