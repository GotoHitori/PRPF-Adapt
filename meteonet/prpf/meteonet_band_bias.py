import torch


def apply_external_band_bias(prediction, biases_dbz, radar_scale=70.0, band_width_dbz=3.0):
    if prediction.ndim != 5 or prediction.shape[2] != 1:
        raise ValueError("prediction must have shape (B,T,1,H,W)")
    biases = prediction.new_tensor(tuple(float(value) for value in biases_dbz))
    if biases.numel() != 3:
        raise ValueError("biases_dbz must contain three values")
    if torch.count_nonzero(biases) == 0:
        return prediction
    raw = prediction * float(radar_scale)
    thresholds = raw.new_tensor((12.0, 24.0, 32.0)).view(1, 1, 3, 1, 1)
    distance = (raw - thresholds).abs()
    gates = torch.sigmoid((float(band_width_dbz) - distance) / 1.0)
    delta = (gates * biases.view(1, 1, 3, 1, 1)).sum(dim=2, keepdim=True)
    corrected = torch.clamp(raw + delta, 0.0, float(radar_scale))
    return corrected / float(radar_scale)
