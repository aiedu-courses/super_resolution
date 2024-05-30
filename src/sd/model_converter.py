# https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/model_converter.py

import torch

from .part_converter.clip import convert_clip_model
from .part_converter.decoder import convert_decoder_model
from .part_converter.diffusion import convert_diffusion_model
from .part_converter.encoder import convert_encoder_model


def load_from_standard_weights(input_file: str, device: str) -> dict[str, torch.Tensor]:
    # Taken from: https://github.com/kjsman/stable-diffusion-pytorch/issues/7#issuecomment-1426839447
    original_model = torch.load(
        input_file,
        map_location=device,
        weights_only=False,
    )["state_dict"]

    converted = {}
    converted["diffusion"] = {}
    converted["encoder"] = {}
    converted["decoder"] = {}
    converted["clip"] = {}

    converted = convert_diffusion_model(converted, original_model)
    converted = convert_encoder_model(converted, original_model)
    converted = convert_clip_model(converted, original_model)
    converted = convert_decoder_model(converted, original_model)

    return converted
