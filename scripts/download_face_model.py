"""Download the official MediaPipe Face Landmarker bundle, never executable code."""

from hashlib import sha256
from pathlib import Path
import urllib.request
import zipfile


URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DESTINATION = Path(__file__).resolve().parents[1] / "models/face/face_landmarker.task"
EXPECTED_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.exists():
        temporary = DESTINATION.with_suffix(".download")
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                temporary.write_bytes(response.read())
            if sha256(temporary.read_bytes()).hexdigest() != EXPECTED_SHA256:
                raise ValueError("Face Landmarker model checksum mismatch")
            with zipfile.ZipFile(temporary) as bundle:
                if "face_landmarks_detector.tflite" not in bundle.namelist():
                    raise ValueError("Unexpected Face Landmarker model bundle")
            temporary.replace(DESTINATION)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(DESTINATION.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Existing Face Landmarker model checksum mismatch")
    print(DESTINATION)
    print("SHA256:", sha256(DESTINATION.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
