import cv2
import os

input_video = r'C:\Users\albed\Desktop\NeuralNetwork\Facial AI Github\dataImages\albe3.MOV'
output_root_directory = r'C:\Users\albed\Desktop\NeuralNetwork\Facial AI Github\dataImages\albe'

# Create the output directory if it doesn't exist
os.makedirs(output_root_directory, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(input_video)

frame_count = 0

# Read and save frames as PNGs
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Define the output file path for the PNG image
    output_file_path = os.path.join(output_root_directory, f'frame_{450 + frame_count:04d}.png')

    # Save the frame as a PNG image
    cv2.imwrite(output_file_path, frame)

# Release the video capture object
cap.release()

print(f"Frames extracted and saved to {output_root_directory}")
