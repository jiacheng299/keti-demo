from scripts.validate_fire_model import build_parser, count_classes


def test_count_classes_uses_model_class_names():
    counts = count_classes(
        class_ids=[0, 1, 1, 0, 1],
        class_names={0: "smoke", 1: "fire"},
    )

    assert counts == {"smoke": 2, "fire": 3}


def test_validation_cli_accepts_positive_and_negative_clips():
    args = build_parser().parse_args(
        [
            "--weights",
            "models/fire/fire_smoke_yolov8n.pt",
            "--positive-video",
            "assets/demo_videos/fire_burning.ogv",
            "--negative-video",
            "assets/demo_videos/vtest.avi",
            "--imgsz",
            "416",
        ]
    )

    assert args.positive_video == "assets/demo_videos/fire_burning.ogv"
    assert args.negative_video == "assets/demo_videos/vtest.avi"
    assert args.imgsz == 416
