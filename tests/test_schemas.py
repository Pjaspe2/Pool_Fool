from pool_fool.shared.schemas import OverlayMessage, ShotGuide


def test_overlay_roundtrip():
    shot = ShotGuide(
        valid=True,
        cue_mm=[1.0, 2.0],
        ghost_mm=[3.0, 4.0],
        object_mm=[5.0, 6.0],
        object_index=0,
    )
    msg = OverlayMessage(timestamp_ms=1, stationary=True, shot=shot)
    raw = msg.to_json()
    back = OverlayMessage.from_json(raw)
    assert back.shot.valid
    assert back.shot.cue_mm == [1.0, 2.0]
