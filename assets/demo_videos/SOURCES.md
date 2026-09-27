# Local demonstration videos

These clips were generated locally from the original sources; they are not byte-for-byte copies of the absent demonstration clips used by the repository author.

- `vtest.avi`: OpenCV sample, https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi (Apache-2.0, per project documentation). `border_demo.avi` uses the first 120 frames, resized from 768x576 to 640x480, MJPEG AVI at 10 FPS.
- `fire_burning.ogv`: "Fire burning", by Zouavman Le Zouave, https://commons.wikimedia.org/wiki/File:Fire_burning.ogv . License: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ . `fire_demo.avi` uses the first 275 frames at 640x480, transcoded to MJPEG AVI at 25 FPS, with audio removed. This derived clip retains CC BY-SA 4.0.

Original SHA-256 values were checked against docs/model_selection.md. Download and transformation reports are in runs/asset-download-report.json and runs/prepared-videos-report.json.

## Monitoring examples added 2026-09-27

- `on_duty_demo.mp4`: CDnet `baseline/office`, https://changedetection.net/dataset2014/ . Download: https://changedetection.net/static/dataset/baseline/office.zip . All 2050 JPEG input frames in their original order, encoded as H.264/yuv420p at 360x240. Playback is set to 25 FPS for this demonstration (82 seconds); the source acquisition frame rate was not verified. Source archive: `runs/office_source.zip`. Retain CDnet attribution and cite its paper when reporting research results.
- `drowsiness_demo.mp4`: unchanged `data/video2.mp4` from https://github.com/incluit/OpenVino-Driver-Behaviour . Direct source: https://raw.githubusercontent.com/incluit/OpenVino-Driver-Behaviour/master/data/video2.mp4 . Shows open/closed-eye actions; not clinically labeled sleep ground truth.
- `head_turn_control.mp4`: unchanged `data/video3.mp4` from the same repository. Direct source: https://raw.githubusercontent.com/incluit/OpenVino-Driver-Behaviour/master/data/video3.mp4 . Head-turn control clip. The repository declares Apache-2.0; its license is retained in `OpenVino-Driver-Behaviour-LICENSE.txt`. No separate likeness release has been verified.

All three files passed complete OpenCV decoding. Exact dimensions, frame counts, byte sizes, hashes and source URLs are recorded in `monitoring_samples.json`. Both offline templates are now implemented and verified end-to-end; design and measured results: `docs/07_duty_and_drowsiness_design.md`.

## Multi-person, lower-resolution samples added 2026-09-27

- `multi_person_360p.mp4`: 20-second excerpt (00:50–01:10) from the 640×360 archived version of **Interview Wikivoyage ohne Abspann.webm**, showing three seated interviewees. Source page: https://commons.wikimedia.org/wiki/File:Interview_Wikivoyage_ohne_Abspann.webm . Downloaded archive: https://upload.wikimedia.org/wikipedia/commons/archive/c/cb/20140621115238%21Interview_Wikivoyage_ohne_Abspann.webm . Attribution: WikiTV, user: .js (host), Flo Sorg (camera), Björn Bauhofer (editor), Conrad Nutschan (data processing). Licensed CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/ . Changes: trimmed to 20 seconds, removed audio, resampled using source timestamps to constant 25 FPS, encoded H.264/yuv420p CRF23. The source frame was not artificially blurred. This derivative retains CC BY-SA 3.0.
- `multi_person_240p.mp4`: the same excerpt resized to 426×240 and encoded at CRF30 for a stronger resolution/compression challenge. Same attribution and CC BY-SA 3.0 license. This is an intentionally degraded derivative, not a separate camera recording.

These clips are multi-person / reduced face-detail boundary tests, not labeled drowsiness-positive examples. The initial single-person limitation has now been removed, but small faces or unclear eyes still cannot be measured reliably. Original source retained at `runs/multi_person_source/wikivoyage_360p.webm`; preparation and full-pipeline verification reports are in that directory. Both outputs were fully decoded: 500 frames, 25 FPS, 20 seconds each. Output hashes and sizes are in `monitoring_samples.json`.

## Clear multi-face samples added 2026-09-27

- `multi_person_clear_1080p.mp4`: cottonbro studio, **Two People Looking at the Camera**, https://www.pexels.com/video/two-people-looking-at-the-camera-6559938/ . Pexels License: https://www.pexels.com/license/ . Download: https://videos.pexels.com/video-files/6559938/6559938-uhd_4096_2160_25fps.mp4 . Changes: first 20 seconds, resized from 4096x2160 to 1920x1012 preserving aspect ratio, removed audio, H.264 CRF18. Real two-person eye movements, not medically labeled sleep footage. No endorsement implied. Original retained in `runs/multi_person_source/pexels_6559938.mp4`.
- `multi_face_staggered_composite.mp4`: two copies of the same actor from the Apache-2.0 OpenVino-Driver-Behaviour `data/video2.mp4` source listed above. Right stream begins 3 seconds ahead and freezes its last frame to match the left stream duration. Original 540x960 pixels per stream retained; a 48-pixel header visibly labels the composite. Audio removed, H.264 CRF18, total 1080x1008, 20 FPS, 203 frames. This is a temporal-rule fixture, NOT footage of two different people in one scene.

Preparation script: `scripts/prepare_multiface_samples.py`; full-video verification: `runs/multi_person_source/clear_verification.json`. Both new clips fully decoded; exact hashes recorded in `monitoring_samples.json`.
