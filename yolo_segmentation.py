from ultralytics import YOLO
import cv2
import os

def yolo_object_segmentation(input_video_path, output_dir, model_path="yolov8n-seg.pt"):
    """
    Perform object segmentation using YOLO and save results to a specified directory.
    
    Args:
        input_video_path (str): Path to the input video file.
        output_dir (str): Directory where segmented frames and data will be saved.
        model_path (str): Path to the YOLO model weights file. Defaults to 'yolov8n-seg.pt'.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open the video file. Check the path.")
        return

    frame_idx = 0

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        print(f"Processing frame {frame_idx}...")  # Debugging

        # Perform inference
        results = model(frame)

        # Debugging: Check if results are obtained
        if results and results[0].boxes:
            print(f"Frame {frame_idx}: {len(results[0].boxes)} objects detected.")
        else:
            print(f"Frame {frame_idx}: No objects detected.")
            frame_idx += 1
            continue

        # Save the segmented frame
        result_img = results[0].plot()  # Get the annotated image
        output_frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_frame_path, result_img)

       # Save detection data (e.g., bounding boxes, masks)
        output_data_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.txt")
        with open(output_data_path, "w") as f:
            for obj in results[0].boxes:
                label = model.names[int(obj.cls)]
                # Convert tensor to float for bounding box coordinates
                x1, y1, x2, y2 = [coord.item() for coord in obj.xyxy[0]]
                confidence = obj.conf.item()  # Convert confidence tensor to float
                f.write(f"{label},{confidence:.4f},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n")

        
        frame_idx += 1

    # Release the video capture
    cap.release()
    print(f"Segmentation completed. Results saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Define paths
    input_video = r"D:\Maharashta Drone Mission\videotrail.mp4"  # Replace with your video file path
    output_directory = r"D:\Maharashta Drone Mission"
    
    # Run the segmentation
    yolo_object_segmentation(input_video, output_directory)
