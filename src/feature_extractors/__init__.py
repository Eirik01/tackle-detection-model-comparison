"""
Feature Extractors Module
Provides backbone-agnostic feature extraction for action spotting.
"""

from .base_extractor import BaseFeatureExtractor
from .dinov3_extractor import DINOv3Extractor
from .vjepa2_extractor import VJEPA2Extractor

__all__ = ['BaseFeatureExtractor', 'DINOv3Extractor', 'VJEPA2Extractor']
