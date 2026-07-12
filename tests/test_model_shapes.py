import torch

from radar2dcnn.model import Radar2DCNN, RadarDSCNN2D, load_checkpoint


def test_model_output_shape():
    model = Radar2DCNN(num_classes=4)
    x = torch.randn(2, 1, 100, 100)
    y = model(x)
    assert y.shape == (2, 4)


def test_dscnn_output_shape_and_operator_constraints():
    model = RadarDSCNN2D(num_classes=4)
    y = model(torch.randn(2, 1, 100, 100))
    assert y.shape == (2, 4)
    assert not any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert any(
        isinstance(module, torch.nn.Conv2d)
        and module.groups == module.in_channels
        and module.in_channels > 1
        for module in model.modules()
    )


def test_dscnn_checkpoint_reloads_correct_architecture(tmp_path):
    model = RadarDSCNN2D(num_classes=4)
    checkpoint = tmp_path / "dscnn.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "architecture": "radar_dscnn2d",
                "num_classes": 4,
            },
        },
        checkpoint,
    )
    loaded = load_checkpoint(str(checkpoint))
    assert isinstance(loaded, RadarDSCNN2D)
    assert loaded(torch.randn(1, 1, 100, 100)).shape == (1, 4)
