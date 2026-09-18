import pytest

from src.video.frame_sampler import FrameSampler


def test_frame_sampler_processes_frames_at_the_configured_interval():
    sampler = FrameSampler(every_n_frames=3)

    decisions = [sampler.should_process(frame_id) for frame_id in range(7)]

    assert decisions == [True, False, False, True, False, False, True]


def test_frame_sampler_rejects_a_non_positive_interval():
    with pytest.raises(ValueError, match="every_n_frames"):
        FrameSampler(every_n_frames=0)


def test_frame_sampler_rejects_a_negative_frame_id():
    sampler = FrameSampler(every_n_frames=2)

    with pytest.raises(ValueError, match="frame_id"):
        sampler.should_process(-1)
