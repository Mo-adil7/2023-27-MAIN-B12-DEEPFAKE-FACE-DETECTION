import csv
import cv2
import random
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("prepared_dataset")
FRAME_COUNT = 8
IMAGE_SIZE = 224
SEED = 42

# Labels inferred from source folder names: REAL=1, FAKE=0.
SOURCE_FOLDERS = {
    "Celeb-real": 1,
    "YouTube-real": 1,
    "Celeb-Youtube-fake": 0,
}


def extract_frames(video_path):
    """Extract 8 evenly spaced RGB frames and center-crop to 224x224."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"WARNING: Cannot open {video_path}")
        return None

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_total < 1:
        cap.release()
        print(f"WARNING: Invalid frame count: {video_path}")
        return None

    indices = [
        round(i * (frame_total - 1) / (FRAME_COUNT - 1))
        for i in range(FRAME_COUNT)
    ]
    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()

        if not ok or frame is None:
            cap.release()
            print(f"WARNING: Cannot read frame {index}: {video_path}")
            return None

        height, width = frame.shape[:2]
        scale = 256 / min(height, width)
        new_width = round(width * scale)
        new_height = round(height * scale)

        frame = cv2.resize(
            frame, (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        left = (new_width - IMAGE_SIZE) // 2
        top = (new_height - IMAGE_SIZE) // 2
        frame = frame[top:top + IMAGE_SIZE, left:left + IMAGE_SIZE]

        if frame.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            cap.release()
            print(f"WARNING: Incorrect frame dimensions: {video_path}")
            return None

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames if len(frames) == FRAME_COUNT else None


def assign_splits(records):
    """Create reproducible stratified splits at the video level."""
    rng = random.Random(SEED)
    split_for_video = {}

    for label in (0, 1):
        group = [r for r in records if r["label"] == label]
        rng.shuffle(group)

        count = len(group)
        if count < 3:
            raise ValueError(
                f"Need at least 3 usable videos for label {label}; got {count}."
            )

        train_end = int(count * 0.70)
        val_end = int(count * 0.85)

        # Ensure all three splits have at least one video per class.
        train_end = max(1, min(train_end, count - 2))
        val_end = max(train_end + 1, min(val_end, count - 1))

        for r in group[:train_end]:
            split_for_video[r["video_id"]] = "train"
        for r in group[train_end:val_end]:
            split_for_video[r["video_id"]] = "validation"
        for r in group[val_end:]:
            split_for_video[r["video_id"]] = "test"

    return split_for_video


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Rename or remove it before "
            "running this script; existing output will not be overwritten."
        )

    source_videos = {}
    for folder_name in SOURCE_FOLDERS:
        folder = DATA_DIR / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Required dataset folder is missing: {folder}")

        source_videos[folder_name] = sorted(folder.glob("*.mp4"))
        print(f"{folder_name}: {len(source_videos[folder_name])} videos")

    OUTPUT_DIR.mkdir(parents=True)
    records = []
    failures = []

    for folder_name, label in SOURCE_FOLDERS.items():
        videos = source_videos[folder_name]
        print(f"\nProcessing {folder_name} ({len(videos)} videos)")

        for number, video_path in enumerate(videos, start=1):
            frames = extract_frames(video_path)

            if frames is None:
                failures.append(str(video_path))
                continue

            video_id = f"{folder_name}_{video_path.stem}"
            video_dir = OUTPUT_DIR / video_id
            video_dir.mkdir(parents=True, exist_ok=False)

            success = True
            for frame_number, rgb_frame in enumerate(frames):
                output_path = video_dir / f"frame_{frame_number:02d}.png"
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                if not cv2.imwrite(str(output_path), bgr_frame):
                    success = False
                    break

            if not success:
                failures.append(str(video_path))
                for path in video_dir.glob("*"):
                    path.unlink()
                video_dir.rmdir()
                continue

            records.append({
                "video_id": video_id,
                "source": folder_name,
                "video_path": str(video_path),
                "label": label,
                "label_name": "REAL" if label == 1 else "FAKE",
                "frames_dir": str(video_dir),
            })

            if number % 100 == 0:
                print(f"  Checked {number}/{len(videos)} videos")

        print(f"  Usable videos so far: {len(records)}")

    if not records:
        raise RuntimeError("No videos were successfully prepared.")

    # Assign splits per video; never split individual frames across sets.
    split_for_video = assign_splits(records)

    manifest_path = OUTPUT_DIR / "manifest.csv"
    fields = [
        "video_id", "source", "video_path",
        "label", "label_name", "frames_dir", "split",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for record in records:
            record["split"] = split_for_video[record["video_id"]]
            writer.writerow(record)

    print("\nDataset preparation complete.")
    print(f"Prepared videos: {len(records)}")
    print(f"Failed videos: {len(failures)}")
    print(f"Manifest: {manifest_path}")
    print("Split counts:")

    for split_name in ("train", "validation", "test"):
        group = [r for r in records if r["split"] == split_name]
        real_count = sum(r["label"] == 1 for r in group)
        fake_count = sum(r["label"] == 0 for r in group)
        print(
            f"  {split_name}: {len(group)} videos "
            f"(REAL={real_count}, FAKE={fake_count})"
        )

    if failures:
        with (OUTPUT_DIR / "failed_videos.txt").open(
            "w", encoding="utf-8"
        ) as file:
            file.write("\n".join(failures) + "\n")
        print(f"Failed video list: {OUTPUT_DIR / 'failed_videos.txt'}")

    print("\nThe original data/test folder was not used.")
    print("Labels are inferred from folder names and should be verified.")


if __name__ == "__main__":
    main()
