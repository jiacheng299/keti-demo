"""Prepare clear real two-person footage and a labeled temporal-rule fixture."""

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/multi_person_source"
OUT = ROOT / "assets/demo_videos"
PEXELS_PAGE = "https://www.pexels.com/video/two-people-looking-at-the-camera-6559938/"
PEXELS_DOWNLOAD = "https://videos.pexels.com/video-files/6559938/6559938-uhd_4096_2160_25fps.mp4"
DRIVER_DOWNLOAD = "https://raw.githubusercontent.com/incluit/OpenVino-Driver-Behaviour/master/data/video2.mp4"


def fetch(url, path):
    if not path.is_file():
        with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as file:
            while block := response.read(1024 * 1024):
                file.write(block)


def verify(path):
    capture = cv2.VideoCapture(str(path))
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if count == 0:
            cv2.imencode(".jpg", frame)[1].tofile(str(SOURCE / (path.stem + "_preview.jpg")))
        count += 1
    capture.release()
    if count != expected or count == 0:
        raise RuntimeError(f"Incomplete video: {path.name}, {count}/{expected}")
    return dict(file=path.name, width=width, height=height, fps=fps, frames=count,
                duration_seconds=count / fps, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), fully_decoded=True)


def main():
    SOURCE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    original = SOURCE / "pexels_6559938.mp4"
    fetch(PEXELS_DOWNLOAD, original)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clear = OUT / "multi_person_clear_1080p.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(original), "-t", "20",
                    "-an", "-vf", "fps=25,scale=1920:-2:flags=lanczos,setsar=1", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clear)], check=True)

    driver = OUT / "drowsiness_demo.mp4"
    fetch(DRIVER_DOWNLOAD, driver)
    left, right = cv2.VideoCapture(str(driver)), cv2.VideoCapture(str(driver))
    right.set(cv2.CAP_PROP_POS_FRAMES, 60)  # 3 seconds ahead; source has 20 FPS.
    composite = OUT / "multi_face_staggered_composite.mp4"
    process = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", "1080x1008", "-r", "20", "-i", "-", "-an", "-c:v", "libx264",
                                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(composite)],
                               stdin=subprocess.PIPE)
    last_right = None
    try:
        while True:
            ok, a = left.read()
            if not ok:
                break
            ok, b = right.read()
            if ok:
                last_right = b
            if last_right is None or a.shape != (960, 540, 3):
                raise RuntimeError("Unexpected source geometry")
            frame = np.zeros((1008, 1080, 3), np.uint8)
            frame[48:, :540], frame[48:, 540:] = a, last_right
            cv2.putText(frame, "COMPOSITE TEST: same actor twice | Right +3s (last frame held at end)",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
            process.stdin.write(frame.tobytes())
    finally:
        left.release()
        right.release()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Composite encoding failed")
    report = [
        dict(**verify(clear), source=PEXELS_PAGE, download_url=PEXELS_DOWNLOAD,
             author="cottonbro studio", license="Pexels License", license_url="https://www.pexels.com/license/",
             changes="First 20 seconds; resized 4096x2160 to 1920x1012; audio removed; H.264 CRF18",
             purpose="Real two-person clear-eye observation, not labeled drowsiness ground truth"),
        dict(**verify(composite), source=DRIVER_DOWNLOAD, license="Apache-2.0",
             changes="Same actor in two simultaneous panels; source pixels retained; right stream 3s ahead; final right frame held for 3s; visible composite label; audio removed",
             purpose="Independent track timing and recovery fixture; NOT a real two-person recording")]
    manifest = OUT / "monitoring_samples.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    names = {r["file"] for r in report}
    manifest.write_text(json.dumps([r for r in previous if r["file"] not in names] + report,
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    (SOURCE / "clear_preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
