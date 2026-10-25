"""Estimate unique visible face identities using MediaPipe and FaceNet.

Heuristic tracking revision: validate with manually counted videos before
claiming improved accuracy. It does not count people whose faces are unseen.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        return None
    return vector / norm


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def nearby(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    scale = max(math.hypot(a[2] - a[0], a[3] - a[1]),
                math.hypot(b[2] - b[0], b[3] - b[1]), 1)
    return math.hypot(ax - bx, ay - by) / scale


@dataclass
class Identity:
    id: int
    box: tuple
    first_seen: float
    last_seen: float
    gallery: list = field(default_factory=list)
    hits: int = 1
    best_area: int = 0
    best_face: object = None

    def confirmed(self):
        return self.hits >= 3 and self.last_seen - self.first_seen >= 0.5

    def similarity(self, embedding):
        return max(float(np.dot(embedding, ref)) for ref in self.gallery)


class PersonCounter:
    def __init__(self):
        self.identities = []
        self.peak_faces = 0

    def update(self, observations, timestamp):
        """Observations have box, embedding and an optional RGB face crop."""
        self.peak_faces = max(self.peak_faces, len(observations))
        candidates = []
        for index, obs in enumerate(observations):
            for identity in self.identities:
                similarity = identity.similarity(obs['embedding'])
                gap = timestamp - identity.last_seen
                overlap = iou(obs['box'], identity.box)
                distance = nearby(obs['box'], identity.box)
                local = (0 < gap <= 1.0 and overlap >= 0.15 and distance <= 0.75)
                # A short, spatially continuous track can tolerate a pose change.
                # Some appearance evidence is still required. Across gaps, use
                # the original global similarity threshold.
                if similarity >= 0.65 or (local and similarity >= 0.40):
                    score = similarity + (0.2 * overlap if local else 0)
                    candidates.append((score, index, identity.id, similarity))

        matches, used_ids = {}, set()
        for score, index, person_id, similarity in sorted(candidates, reverse=True):
            if index not in matches and person_id not in used_ids:
                matches[index] = (self.identities[person_id - 1], similarity)
                used_ids.add(person_id)

        assignments = []
        for index, obs in enumerate(observations):
            if index in matches:
                identity, similarity = matches[index]
                identity.box = obs['box']
                identity.last_seen = timestamp
                identity.hits += 1
                # Avoid adding very weak continuity matches to the face gallery.
                if similarity >= 0.55:
                    identity.gallery.append(obs['embedding'].copy())
                    if len(identity.gallery) > 12:
                        # Preserve the first reference plus recent appearances.
                        identity.gallery.pop(1)
            else:
                identity = Identity(
                    len(self.identities) + 1, obs['box'], timestamp, timestamp,
                    gallery=[obs['embedding'].copy()],
                )
                self.identities.append(identity)
            area = (obs['box'][2] - obs['box'][0]) * (obs['box'][3] - obs['box'][1])
            if obs.get('face') is not None and area > identity.best_area:
                identity.best_area = area
                identity.best_face = obs['face'].copy()
            assignments.append(identity)
        return assignments


def counting_region(frame, grid=1):
    """Select one whole tile only when the user declares a repeated scene.

    No identity count is imposed: multiple visible people inside the tile
    still pass through the usual detector and tracker.
    """
    if grid not in (1, 3):
        raise ValueError('Supported counting layouts are standard and repeated 3x3.')
    height, width = frame.shape[:2]
    if min(height, width) < grid:
        raise ValueError('Video dimensions are too small for this grid.')
    box = (0, 0, width // grid, height // grid)
    if grid == 1:
        return frame, box
    return frame[:height // grid, :width // grid], box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--repeated-grid', type=int, choices=(1, 3), default=1,
                        help='3: count in the upper-left tile of a declared repeated 3x3 scene.')
    args = parser.parse_args()

    import cv2
    import mediapipe as mp
    from keras_facenet import FaceNet

    video_path = Path(args.video).resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')

    try:
        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(raw_count) if np.isfinite(raw_count) else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if np.isfinite(fps) and fps > 0 else 0.0
        timing_fps = fps or 25.0
        sample_every = max(1, round(timing_fps / 4))
        output = Path(__file__).resolve().parent.parent / 'detected_faces' / video_path.stem
        output.mkdir(parents=True, exist_ok=True)
        print('Loading FaceNet...', flush=True)
        embedder = FaceNet()
        counter = PersonCounter()
        frame_count = samples = face_detections = 0
        stopped_early = False
        region_box = None
        source_dimensions = None

        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as detector:
            while True:
                ok, original = cap.read()
                if not ok:
                    break
                frame_count += 1
                if frame_count != 1 and (frame_count - 1) % sample_every:
                    continue
                samples += 1
                source_dimensions = {'width': original.shape[1], 'height': original.shape[0]}
                original, region_box = counting_region(original, args.repeated_grid)
                h, w = original.shape[:2]
                scale = min(1280 / max(h, w), 1.0)
                small = cv2.resize(original, (max(1, round(w * scale)), max(1, round(h * scale))))
                sh, sw = small.shape[:2]
                detected = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                boxes = []
                for detection in detected.detections or []:
                    bb = detection.location_data.relative_bounding_box
                    box = (max(0, int(bb.xmin * w)), max(0, int(bb.ymin * h)),
                           min(w, int((bb.xmin + bb.width) * w)),
                           min(h, int((bb.ymin + bb.height) * h)))
                    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                        continue
                    boxes.append((float(detection.score[0]), box))
                # Suppress duplicate detector boxes before matching identities.
                kept = []
                for confidence, box in sorted(boxes, reverse=True):
                    if all(iou(box, old) < 0.45 for old in kept):
                        kept.append(box)

                observations, faces = [], []
                for box in kept:
                    x1, y1, x2, y2 = box
                    crop = cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    faces.append(cv2.resize(crop, (160, 160)))
                if faces:
                    embeddings = embedder.embeddings(faces)
                    if len(embeddings) != len(faces):
                        raise ValueError('FaceNet returned an unexpected embedding count.')
                    for box, face, vector in zip(kept, faces, embeddings):
                        embedding = unit(vector)
                        if embedding is not None:
                            observations.append(dict(box=box, embedding=embedding, face=face))
                face_detections += len(observations)
                assigned = counter.update(observations, (frame_count - 1) / timing_fps)

                if args.preview:
                    for observation, identity in zip(observations, assigned):
                        x1, y1, x2, y2 = observation['box']
                        a, b, c, d = round(x1 * sw / w), round(y1 * sh / h), round(x2 * sw / w), round(y2 * sh / h)
                        color = (100, 220, 100) if identity.confirmed() else (50, 180, 240)
                        cv2.rectangle(small, (a, b), (c, d), color, 2)
                        label = f"ID {identity.id}" + ('' if identity.confirmed() else ' tentative')
                        cv2.putText(small, label, (a, max(18, b - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                    cv2.imshow('Face identity estimate', small)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stopped_early = True
                        break
                if samples % 4 == 0:
                    progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                    print(f'PROGRESS:{frame_count}:{total_frames}:{progress:.1f}', flush=True)

        if frame_count == 0:
            raise ValueError('No video frames could be read.')
        confirmed = [p for p in counter.identities if p.confirmed()]
        diagnostics = []
        for identity in counter.identities:
            crop_path = None
            if identity.best_face is not None:
                candidate_path = output / f'identity_{identity.id:03d}.jpg'
                if cv2.imwrite(str(candidate_path), cv2.cvtColor(identity.best_face, cv2.COLOR_RGB2BGR)):
                    crop_path = str(candidate_path)
            diagnostics.append({
                'id': identity.id, 'confirmed': identity.confirmed(),
                'observations': identity.hits,
                'first_seen_seconds': round(identity.first_seen, 2),
                'last_seen_seconds': round(identity.last_seen, 2),
                'representative_crop': crop_path,
            })
        result = {
            'total_frames': total_frames, 'frames_read': frame_count,
            'faces_detected': len(confirmed),  # Compatibility with current app.
            'unique_people_estimate': len(confirmed),
            'face_detections_used': face_detections,
            'max_faces_in_sampled_frame': counter.peak_faces,
            'unconfirmed_identity_candidates': len(counter.identities) - len(confirmed),
            'sampled_frames': samples, 'fps': fps,
            'video_length': round(total_frames / fps, 2) if fps else None,
            'stopped_early': stopped_early,
            'counting_layout': 'repeated_3x3' if args.repeated_grid == 3 else 'standard',
            'counting_region': {'tile': 'upper-left' if args.repeated_grid == 3 else 'full-frame',
                                'box_xyxy': region_box, 'source_dimensions': source_dimensions},
            'identity_diagnostics': diagnostics,
            'count_note': 'Estimated distinct visible face identities. Faces need at least 3 matched observations spanning 0.5 seconds. Brief, hidden or difficult faces may be missed; one person may still split into several identities.',
            'diagnostics_folder': str(output),
        }
        if args.repeated_grid == 3:
            result['count_note'] = (
                'Repeated 3x3 mode was selected: the people estimate uses only the upper-left tile. '
                'Use this mode only when all tiles repeat the same scene. The count is inferred '
                'from faces in that tile; it is not fixed to one. Brief or hidden faces may be missed.'
            )
        (output / 'identity_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'Estimated unique people: {len(confirmed)}', flush=True)
        print(json.dumps(result), flush=True)  # Flask requires JSON on the LAST line.
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Face analysis failed: {exc}', file=sys.stderr)
        sys.exit(1)
