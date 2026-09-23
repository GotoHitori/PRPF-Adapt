import torch


def apply_confidence_band_bias(
    prediction,
    probability,
    static_bias_dbz,
    modulation_dbz=None,
    radar_scale=70.0,
    band_width_dbz=3.0,
):
    if prediction.ndim != 5 or prediction.shape[2] != 1:
        raise ValueError("prediction must have shape (B,T,1,H,W)")
    batch, leads, _, height, width = prediction.shape
    expected = (batch, leads, 3, height, width)
    if tuple(probability.shape) != expected:
        raise ValueError(f"probability must have shape {expected}")
    static = prediction.new_tensor(tuple(float(value) for value in static_bias_dbz))
    if static.numel() != 3:
        raise ValueError("static_bias_dbz must contain three values")
    if modulation_dbz is None:
        modulation = torch.zeros_like(static)
    else:
        modulation = prediction.new_tensor(tuple(float(value) for value in modulation_dbz))
        if modulation.numel() != 3:
            raise ValueError("modulation_dbz must contain three values")
    raw = prediction * float(radar_scale)
    thresholds = raw.new_tensor((12.0, 24.0, 32.0)).view(1, 1, 3, 1, 1)
    gates = torch.sigmoid((float(band_width_dbz) - (raw - thresholds).abs()) / 1.0)
    bias = static.view(1, 1, 3, 1, 1) + modulation.view(1, 1, 3, 1, 1) * (probability.to(raw.dtype) - 0.5)
    delta = (gates * bias).sum(dim=2, keepdim=True)
    return torch.clamp(raw + delta, 0.0, float(radar_scale)) / float(radar_scale)
