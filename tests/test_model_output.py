from scripts.validate_border_model import build_parser, normalize_detections


def test_normalize_detections_converts_box_and_class_metadata():
    detections = normalize_detections(
        boxes=[[10, 20, 30, 40]],
        confidences=[0.875],
        class_ids=[0],
        class_names={0: "person"},
        frame_id=7,
    )

    assert detections == [
        {
            "frame_id": 7,
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.875,
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
        }
    ]


def test_validation_cli_accepts_required_baseline_options():
    args = build_parser().parse_args(
        [
            "--video",
            "sample.mp4",
            "--weights",
            "models/border/yolo11n.pt",
            "--imgsz",
            "416",
        ]
    )

    assert args.video == "sample.mp4"
    assert args.weights == "models/border/yolo11n.pt"
    assert args.imgsz == 416
