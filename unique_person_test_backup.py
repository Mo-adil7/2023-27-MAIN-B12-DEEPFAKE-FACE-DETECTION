import cv2
import mediapipe as mp
import numpy as np
import sys
import os
from keras_facenet import FaceNet

# ==================================================
# SETTINGS
# ==================================================

VIDEO_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "uploads/7647241-uhd_3840_2160_24fps.mp4"
)

RESIZE_WIDTH = 640

# Starting value only; must be evaluated on test videos.
MATCH_THRESHOLD = 0.75

# Maximum number of embeddings stored for each identity.
MAX_EMBEDDINGS_PER_PERSON = 10

# Maximum gap for retaining a recent face track.
MAX_MISSED_FRAMES = 30

# Process every frame for this initial test.
FRAME_STEP = 1

# ==================================================
# LOAD FACENET
# ==================================================

print("Loading FaceNet model...")

embedder = FaceNet()

print("FaceNet loaded successfully.")

# ==================================================
# OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_length = total_frames / fps if fps > 0 else 0

print(f"Total frames: {total_frames}")
print(f"Video FPS: {fps:.2f}")
print(f"Video duration: {video_length:.2f} seconds")

# ==================================================
# FACE DETECTION
# ==================================================

mp_face_detection = mp.solutions.face_detection

# Each identity contains several facial embeddings.
known_people = []

# Recent tracks help associate nearby detections.
tracks = {}

frame_number = 0
face_detections = 0


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def cosine_similarity(a, b):
    """Calculate similarity between normalized embeddings."""

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return -1.0

    return float(np.dot(a, b) / denominator)


def get_embedding(frame, box):
    """Extract a FaceNet embedding from a detected face."""

    x1, y1, x2, y2 = box

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    # FaceNet expects RGB input.
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = cv2.resize(face, (160, 160))

    embedding = embedder.embeddings([face])[0]

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm


def match_person(embedding):
    """Find the best matching known identity."""

    best_person = None
    best_similarity = -1.0

    for person in known_people:

        # Compare against every stored appearance.
        similarities = [
            cosine_similarity(embedding, stored_embedding)
            for stored_embedding in person["embeddings"]
        ]

        if not similarities:
            continue

        similarity = max(similarities)

        if similarity > best_similarity:
            best_similarity = similarity
            best_person = person

    if (
        best_person is not None
        and best_similarity >= MATCH_THRESHOLD
    ):
        return best_person

    return None


def add_embedding(person, embedding):
    """Store a new appearance without unlimited growth."""

    person["embeddings"].append(embedding.copy())

    if len(person["embeddings"]) > MAX_EMBEDDINGS_PER_PERSON:
        person["embeddings"].pop(0)


def box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def center_distance(box_a, box_b):
    """Measure distance between two face-box centers."""

    center_a = box_center(box_a)
    center_b = box_center(box_b)

    return np.sqrt(
        (center_a[0] - center_b[0]) ** 2
        + (center_a[1] - center_b[1]) ** 2
    )


# ==================================================
# PROCESS VIDEO
# ==================================================

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
) as detector:

    while True:

        ret, original_frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Optional frame skipping, disabled for this test.
        if FRAME_STEP > 1 and frame_number % FRAME_STEP != 0:
            continue

        original_height, original_width = original_frame.shape[:2]

        scale = min(
            RESIZE_WIDTH / original_width,
            1.0
        )

        frame = cv2.resize(
            original_frame,
            (
                int(original_width * scale),
                int(original_height * scale)
            )
        )

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        detection_results = detector.process(rgb_frame)

        current_faces = []

        if detection_results.detections:

            for detection in detection_results.detections:

                bbox = (
                    detection.location_data
                    .relative_bounding_box
                )

                x1 = max(0, int(bbox.xmin * width))
                y1 = max(0, int(bbox.ymin * height))

                x2 = min(
                    width,
                    int((bbox.xmin + bbox.width) * width)
                )

                y2 = min(
                    height,
                    int((bbox.ymin + bbox.height) * height)
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                # Ignore very small face crops.
                if x2 - x1 < 40 or y2 - y1 < 40:
                    continue

                current_faces.append((x1, y1, x2, y2))

        # Remove old tracks.
        tracks = {
            person_id: track
            for person_id, track in tracks.items()
            if frame_number - track["last_seen"] <= MAX_MISSED_FRAMES
        }

        # Avoid assigning the same identity to multiple faces
        # in the same frame.
        assigned_this_frame = set()

        # Process larger faces first.
        current_faces.sort(
            key=lambda box: (
                (box[2] - box[0]) * (box[3] - box[1])
            ),
            reverse=True
        )

        for box in current_faces:

            embedding = get_embedding(frame, box)

            if embedding is None:
                continue

            face_detections += 1

            # Step 1: Match using stored facial appearances.
            person = match_person(embedding)

            # Step 2: If there is no embedding match, try a
            # recent track whose face is close to this detection.
            #
            # This spatial fallback is deliberately limited:
            # position alone cannot establish identity.
            if person is None:

                best_track_id = None
                best_distance = float("inf")

                for person_id, track in tracks.items():

                    if person_id in assigned_this_frame:
                        continue

                    missed = frame_number - track["last_seen"]

                    if missed > MAX_MISSED_FRAMES:
                        continue

                    distance = center_distance(
                        box,
                        track["box"]
                    )

                    # Permit only relatively nearby associations.
                    allowed_distance = 100 + missed * 8

                    if (
                        distance < allowed_distance
                        and distance < best_distance
                    ):
                        best_distance = distance
                        best_track_id = person_id

                if best_track_id is not None:

                    person = next(
                        (
                            p for p in known_people
                            if p["id"] == best_track_id
                        ),
                        None
                    )

            # Step 3: Create a new identity only if no match
            # was found by either method.
            if person is None:

                person = {
                    "id": len(known_people) + 1,
                    "embeddings": [embedding.copy()]
                }

                known_people.append(person)

            else:
                add_embedding(person, embedding)

            person_id = person["id"]

            assigned_this_frame.add(person_id)

            # Update recent tracking information.
            tracks[person_id] = {
                "box": box,
                "last_seen": frame_number
            }

        if frame_number % 50 == 0:

            print(
                f"Processed {frame_number}/{total_frames} frames; "
                f"estimated unique people: {len(known_people)}"
            )

cap.release()

# ==================================================
# FINAL RESULTS
# ==================================================

print("\nRecognition test completed.")
print(f"Frames processed: {frame_number}")
print(f"Face detections used: {face_detections}")
print(f"Estimated unique people: {len(known_people)}")
print(f"Video FPS: {fps:.2f}")
print(f"Video duration: {video_length:.2f} seconds")

print(
    "\nNote: This is an experimental estimate, "
    "not a verified unique-person count."
)