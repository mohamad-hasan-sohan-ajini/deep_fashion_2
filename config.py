"""Gather all configuration regarding data, model, and training"""

from dataclasses import dataclass

landmark_indices = {
    1: (0, 25),
    2: (25, 58),
    3: (58, 89),
    4: (89, 128),
    5: (128, 143),
    6: (143, 158),
    7: (158, 168),
    8: (168, 182),
    9: (182, 190),
    10: (190, 219),
    11: (219, 256),
    12: (256, 275),
    13: (275, 294),
}


@dataclass
class DataConfig:
    IMAGE_SIZE: int = 256
