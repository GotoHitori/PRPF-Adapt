from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolConfig:
    batch_size: int = 4
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    lambda_adv_f: float = 0.01
    lambda_adv_s: float = 0.01
    w_phys: float = 0.1
    w_smooth: float = 0.05
    ema_decay: float = 0.999
    weight_decay: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    stage1_epochs: int = 5
    stage2_epochs: int = 5
    smoke_steps: int = 100
    smoke_val_steps: int = 100
    stage_lengths_are_assumptions: bool = True


def protocol_defaults():
    return ProtocolConfig()


def balance_weights(phase):
    if int(phase) == 1:
        return (1.0, 2.0, 4.0, 8.0, 12.0, 12.0, 12.0)
    return (1.0, 1.0, 1.5, 2.0, 3.0, 3.0, 3.0)


def validate_protocol_args(args):
    if not getattr(args, "archive_layout", False):
        raise ValueError("protocol requires archive_layout=True")
    if not getattr(args, "smoke", False) and float(getattr(args, "train_fraction", 1.0)) != 1.0:
        raise ValueError("protocol full runs require train_fraction=1.0")
