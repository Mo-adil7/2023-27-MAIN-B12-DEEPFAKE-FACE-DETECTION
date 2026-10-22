import cv2
import mediapipe as mp
import numpy as np
import sys
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
MATCH_THRESHOLD = 0.75
MAX_EMBEDDINGS_PER_PERSON = 10
MAX_MISSED_FRAMES = 30

# ==================================================
# LOAD MODEL
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
# DATA
# ==================================================

known_people = []
tracks = {}

frame_number = 0
face_detections = 0

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return -1.0

    return float(np.dot(a, b) / denominator)


def get_embedding(frame, box):
    x1, y1, x2, y2 = box
    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))

    embedding = embedder.embeddings([face])[0]
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm


def find_matching_person(embedding, excluded_ids):
    best_person = None
    best_similarity = -1.0

    for person in known_people:
        if person["id"] in excluded_ids:
            continue

        similarities = [
            cosine_similarity(embedding, stored)
            for stored in person["embeddings"]
        ]

        if similarities:
            similarity = max(similarities)

            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person

    if best_person is not None and best_similarity >= MATCH_THRESHOLD:
        return best_person, best_similarity

    return None, best_similarity


def update_embeddings(person, embedding):
    person["embeddings"].append(embedding.copy())

    if len(person["embeddings"]) > MAX_EMBEDDINGS_PER_PERSON:
        person["embeddings"].pop(0)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_distance(box_a, box_b):
    a = box_center(box_a)
    b = box_center(box_b)

    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ==================================================
# PROCESS VIDEO
# ==================================================

with mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
) as detector:

    while True:
        ret, original_frame = cap.read()

        if not ret:
            break

        frame_number += 1

        original_height, original_width = original_frame.shape[:2]

        scale = min(RESIZE_WIDTH / original_width, 1.0)

        frame = cv2.resize(
            original_frame,
            (
                max(1, int(original_width * scale)),
                max(1, int(original_height * scale))
            )
        )

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = detector.process(rgb_frame)

        current_faces = []

        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box

                x1 = max(0, int(bbox.xmin * width))
                y1 = max(0, int(bbox.ymin * height))
                x2 = min(width, int((bbox.xmin + bbox.width) * width))
                y2 = min(height, int((bbox.ymin + bbox.height) * height))

                if x2 <= x1 or y2 <= y1:
                    continue

                if x2 - x1 < 40 or y2 - y1 < 40:
                    continue

                current_faces.append((x1, y1, x2, y2))

        # Keep recent tracks for spatial association.
        tracks = {
            person_id: track
            for person_id, track in tracks.items()
            if frame_number - track["last_seen"] <= MAX_MISSED_FRAMES
        }

        # Process larger face crops first.
        current_faces.sort(
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )

        assigned_ids = set()

        for box in current_faces:
            embedding = get_embedding(frame, box)

            if embedding is None:
                continue

            face_detections += 1

            # Compare the face with known identities.
            person, similarity = find_matching_person(
                embedding,
                assigned_ids
            )

            # Diagnostic output: inspect the best available match.
            if similarity >= 0:
                print(
                    f"Frame {frame_number}: "
                    f"best similarity={similarity:.3f}, "
                    f"match={'Person ' + str(person['id']) if person else 'NEW PERSON'}"
                )
            elif not known_people:
                print(
                    f"Frame {frame_number}: "
                    "first detected identity"
                )
            else:
                print(
                    f"Frame {frame_number}: "
                    "no eligible identity to compare"
                )

            # If facial matching fails, check recent track positions.
            # Position alone is never sufficient to establish identity.
            if person is None:
                best_track_id = None
                best_distance = float("inf")

                for person_id, track in tracks.items():
                    if person_id in assigned_ids:
                        continue

                    missed = frame_number - track["last_seen"]
                    distance = center_distance(box, track["box"])
                    allowed_distance = 60 + missed * 5

                    if (
                        missed <= MAX_MISSED_FRAMES
                        and distance < allowed_distance
                        and distance < best_distance
                    ):
                        best_track_id = person_id
                        best_distance = distance

                if best_track_id is not None:
                    candidate = next(
                        (
                            p for p in known_people
                            if p["id"] == best_track_id
                        ),
                        None
                    )

                    if candidate is not None:
                        candidate_similarity = max(
                            cosine_similarity(embedding, stored)
                            for stored in candidate["embeddings"]
                        )

                        if candidate_similarity >= MATCH_THRESHOLD:
                            person = candidate

            # Create a new identity if no match is accepted.
            if person is None:
                person = {
                    "id": len(known_people) + 1,
                    "embeddings": [embedding.copy()]
                }
                known_people.append(person)
                print(
                    f"  -> Created Person {person['id']}"
                )
            else:
                update_embeddings(person, embedding)

            person_id = person["id"]
            assigned_ids.add(person_id)

            tracks[person_id] = {
                "box": box,
                "last_seen": frame_number
            }

        if frame_number % 50 == 0:
            print(
                f"\nProgress: {frame_number}/{total_frames} frames; "
                f"estimated identities: {len(known_people)}\n"
            )

cap.release()

# ==================================================
# RESULTS
# ==================================================

print("\n" + "=" * 45)
print("RECOGNITION TEST COMPLETED")
print("=" * 45)
print(f"Frames processed: {frame_number}")
print(f"Face detections used: {face_detections}")
print(f"Estimated unique people: {len(known_people)}")
print(f"Video FPS: {fps:.2f}")
print(f"Video duration: {video_length:.2f} seconds")
print("=" * 45)

print(
    "\nNote: This is an experimental estimate, "
    "not a verified unique-person count."
)