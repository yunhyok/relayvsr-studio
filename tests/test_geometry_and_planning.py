from relayvsr_studio.engine.geometry import RelayGeometry
from relayvsr_studio.engine.runtime.streaming import plan_tiles


def test_geometry_4x():
    g = RelayGeometry(4)
    assert g.latent_stride == 4 and g.wan_stride == 8 and g.wan_unshuffle == 16 and g.wan_upsample == 4
    assert g.flash_encoder_strides == (2, 2, 1)
    assert g.padded_size(360, 624) == (360, 624)
    assert g.padded_size(361, 625) == (368, 632)
    assert g.wan_tokens(360, 624) == (45, 78)
    assert g.output_size(360, 624) == (1440, 2496)


def test_geometry_2x():
    g = RelayGeometry(2)
    assert g.latent_stride == 8 and g.wan_stride == 16 and g.wan_unshuffle == 8 and g.wan_upsample == 1
    assert g.flash_encoder_strides == (2, 2, 2)
    assert g.padded_size(1080, 1920) == (1088, 1920)
    assert g.wan_tokens(1080, 1920) == (68, 120)


def test_plan_tiles_covers_grid():
    for th, tw, max_tokens, overlap in ((45, 78, 2000, 8), (68, 120, 3000, 8), (135, 240, 4096, 12), (20, 30, 10_000, 8)):
        tiles = plan_tiles(th, tw, max_tokens, overlap)
        covered = [[0] * tw for _ in range(th)]
        for t in tiles:
            assert 0 <= t.y0 < t.y1 <= th and 0 <= t.x0 < t.x1 <= tw
            assert (t.y1 - t.y0) * (t.x1 - t.x0) <= max_tokens or len(tiles) == 1
            for y in range(t.y0, t.y1):
                for x in range(t.x0, t.x1):
                    covered[y][x] += 1
        assert all(all(c >= 1 for c in row) for row in covered)
        if th * tw <= max_tokens:
            assert len(tiles) == 1


def test_plan_tiles_disabled():
    assert len(plan_tiles(100, 100, 0, 8)) == 1
