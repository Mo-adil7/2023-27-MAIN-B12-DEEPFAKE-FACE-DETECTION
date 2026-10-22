import cv2
import mediapipe as mp
import sys
import os
import json
import numpy as np
from keras_facenet import FaceNet

if len(sys.argv) < 2:
    print("Error: No video file provided.")
    sys.exit(1)

video_path = sys.argv[1]
show_preview = "--preview" in sys.argv

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open the video.")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_length = total_frames / fps if fps > 0 else 0

print(f"Total frames in video: {total_frames}")
print(f"Video FPS: {fps}")
print(f"Video length: {video_length:.2f} seconds")

output_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)

embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection

frame_count = 0
# Each person has several representative embeddings.
person_embeddings = []
similarity_threshold = 0.65
max_representatives = 5
sample_every = max(1, int(fps))  # Sample approximately once per second

if show_preview:
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
) as face_detection:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 10 == 0 or frame_count == total_frames:
            progress = frame_count / total_frames * 100 if total_frames else 0
            print(
                f"PROGRESS:{frame_count}:{total_frames}:{progress:.1f}",
                flush=True
            )

        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        # Sample about one frame per second for identity grouping.
        if results.detections and (frame_count == 1 or frame_count % sample_every == 0):
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x1 = max(0, int(bbox.xmin * width))
                y1 = max(0, int(bbox.ymin * height))
                x2 = min(width, int((bbox.xmin + bbox.width) * width))
                y2 = min(height, int((bbox.ymin + bbox.height) * height))

                face = frame[y1:y2, x1:x2]
                if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                    continue

                # FaceNet expects RGB face images.
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (160, 160))
                embedding = embedder.embeddings([face_rgb])[0]
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    continue
                embedding = embedding / norm

                # Compare against all representatives for each person.
                best_person = None
                best_similarity = -1.0

                for person_index, representatives in enumerate(person_embeddings):
                    person_similarity = max(
                        float(np.dot(embedding, representative))
                        for representative in representatives
                    )

                    if person_similarity > best_similarity:
                        best_similarity = person_similarity
                        best_person = person_index

                if best_person is not None and best_similarity >= similarity_threshold:
                    representatives = person_embeddings[best_person]

                    # Keep a few varied examples to help match pose changes.
                    if len(representatives) < max_representatives:
                        representatives.append(embedding.copy())
                    else:
                        # Replace the oldest representative with the new sample.
                        representatives.pop(0)
                        representatives.append(embedding.copy())
                else:
                    person_embeddings.append([embedding.copy()])

                # Draw rectangles only when preview is enabled.
                if show_preview:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_preview:
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
if show_preview:
    cv2.destroyAllWindows()

unique_people = len(person_embeddings)

results_data = {
    "total_frames": total_frames,
    "frames_read": frame_count,
    "faces_detected": unique_people,
    "unique_people_estimate": unique_people,
    "fps": fps,
    "video_length": round(video_length, 2)
}

print("\nFace detection completed!")
print(f"Frames read: {frame_count}")
print(f"Estimated unique people: {unique_people}")
print(json.dumps(results_data))
