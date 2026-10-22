import cv2
import mediapipe as mp
import sys
import os
import json

# Check whether a video path was provided
if len(sys.argv) < 2:
    print("Error: No video file provided.")
    sys.exit()

# Get video path
video_path = sys.argv[1]

# Check whether preview was requested
show_preview = "--preview" in sys.argv

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video.")
    sys.exit()

# Get video information
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps > 0:
    video_length = total_frames / fps
else:
    video_length = 0

print(f"Total frames in video: {total_frames}")
print(f"Video FPS: {fps}")
print(f"Video length: {video_length:.2f} seconds")

# Create folder for detected faces
output_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

frame_count = 0
face_count = 0

# Create the popup window ONLY if preview is enabled
if show_preview:
    cv2.namedWindow(
        "Face Detection",
        cv2.WINDOW_NORMAL
    )

# Start MediaPipe Face Detection
with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
) as face_detection:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Get original frame dimensions
        height, width = frame.shape[:2]

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        results = face_detection.process(rgb_frame)

        # Check if faces were detected
        if results.detections:

            for detection in results.detections:

                # Get bounding box
                bbox = detection.location_data.relative_bounding_box

                # Convert relative coordinates to pixels
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Make sure coordinates are valid
                x = max(0, x)
                y = max(0, y)

                # Prevent the box from going outside the frame
                w = min(w, width - x)
                h = min(h, height - y)

                # Draw green rectangle ONLY for preview
                if show_preview:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                # Crop detected face
                face = frame[y:y + h, x:x + w]

                # Save face if valid
                if face.size > 0:

                    face_filename = os.path.join(
                        output_folder,
                        f"face_{face_count}.jpg"
                    )

                    cv2.imwrite(
                        face_filename,
                        face
                    )

                    face_count += 1

        # Show preview ONLY if checkbox was selected
        if show_preview:

            # Resize ONLY for display
            height, width = frame.shape[:2]

            max_width = 1000
            max_height = 700

            scale = min(
                max_width / width,
                max_height / height,
                1
            )

            display_frame = cv2.resize(
                frame,
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

            # Show video
            cv2.imshow(
                "Face Detection",
                display_frame
            )

            # Press Q to stop preview
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


# Release resources
cap.release()

# Close OpenCV windows ONLY if preview was shown
if show_preview:
    cv2.destroyAllWindows()


# Create results dictionary
results_data = {
    "total_frames": total_frames,
    "frames_read": frame_count,
    "faces_detected": face_count,
    "fps": fps,
    "video_length": round(video_length, 2)
}


# Print normal results
print("\nFace detection completed!")
print(f"Frames read: {frame_count}")
print(f"Faces detected and saved: {face_count}")
print(f"Video FPS: {fps}")
print(f"Video length: {video_length:.2f} seconds")


# IMPORTANT:
# Print JSON on the LAST line so Flask can read it
print(json.dumps(results_data))