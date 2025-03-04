"""
Objective metrics for evaluating voice separation models.

This module provides functions for calculating standard metrics used in
evaluating voice separation performance, including SI-SNRi (Scale-Invariant
Signal-to-Noise Ratio improvement) and SDRi (Signal-to-Distortion Ratio improvement).
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import torch


def calculate_si_snri(
    estimated_sources: Union[np.ndarray, torch.Tensor],
    target_sources: Union[np.ndarray, torch.Tensor],
    mixture: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio improvement (SI-SNRi).
    
    SI-SNRi measures the improvement in SI-SNR between the separated source
    and the original mixture.
    
    Args:
        estimated_sources: Estimated source signals, shape (n_sources, n_samples)
                          or single source of shape (n_samples,)
        target_sources: Target source signals, shape (n_sources, n_samples)
                       or single source of shape (n_samples,)
        mixture: Optional mixture signal, shape (n_samples,). If not provided,
                only SI-SNR (not improvement) will be calculated.
    
    Returns:
        SI-SNRi value(s) in dB. If multiple sources, returns array of values.
    """
    # Convert to numpy if tensors
    if isinstance(estimated_sources, torch.Tensor):
        estimated_sources = estimated_sources.detach().cpu().numpy()
    if isinstance(target_sources, torch.Tensor):
        target_sources = target_sources.detach().cpu().numpy()
    if mixture is not None and isinstance(mixture, torch.Tensor):
        mixture = mixture.detach().cpu().numpy()
    
    # Handle single source case
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if target_sources.ndim == 1:
        target_sources = target_sources[np.newaxis, :]
    
    n_sources = estimated_sources.shape[0]
    si_snr_values = np.zeros(n_sources)
    
    for i in range(n_sources):
        # Calculate SI-SNR for the separated source
        si_snr_values[i] = _calculate_si_snr(estimated_sources[i], target_sources[i])
    
    # If mixture is provided, calculate SI-SNRi
    if mixture is not None:
        si_snr_mixture = np.zeros(n_sources)
        for i in range(n_sources):
            si_snr_mixture[i] = _calculate_si_snr(mixture, target_sources[i])
        
        # SI-SNRi = SI-SNR(estimated, target) - SI-SNR(mixture, target)
        si_snri_values = si_snr_values - si_snr_mixture
        return si_snri_values
    
    return si_snr_values


def _calculate_si_snr(estimated: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR) for a single source.
    
    Args:
        estimated: Estimated source signal, shape (n_samples,)
        target: Target source signal, shape (n_samples,)
    
    Returns:
        SI-SNR value in dB
    """
    # Zero-mean normalization
    estimated = estimated - np.mean(estimated)
    target = target - np.mean(target)
    
    # Scale invariant projection
    s_target = np.dot(estimated, target) * target / np.sum(target**2)
    
    # Error
    e_noise = estimated - s_target
    
    # SI-SNR
    si_snr = 10 * np.log10(np.sum(s_target**2) / np.sum(e_noise**2) + 1e-8)
    
    return si_snr


def calculate_sdri(
    estimated_sources: Union[np.ndarray, torch.Tensor],
    target_sources: Union[np.ndarray, torch.Tensor],
    mixture: Union[np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray]:
    """
    Calculate Signal-to-Distortion Ratio improvement (SDRi).
    
    SDRi measures the improvement in SDR between the separated source
    and the original mixture.
    
    Args:
        estimated_sources: Estimated source signals, shape (n_sources, n_samples)
                          or single source of shape (n_samples,)
        target_sources: Target source signals, shape (n_sources, n_samples)
                       or single source of shape (n_samples,)
        mixture: Mixture signal, shape (n_samples,)
    
    Returns:
        SDRi value(s) in dB. If multiple sources, returns array of values.
    """
    # Convert to numpy if tensors
    if isinstance(estimated_sources, torch.Tensor):
        estimated_sources = estimated_sources.detach().cpu().numpy()
    if isinstance(target_sources, torch.Tensor):
        target_sources = target_sources.detach().cpu().numpy()
    if isinstance(mixture, torch.Tensor):
        mixture = mixture.detach().cpu().numpy()
    
    # Handle single source case
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if target_sources.ndim == 1:
        target_sources = target_sources[np.newaxis, :]
    
    n_sources = estimated_sources.shape[0]
    sdr_values = np.zeros(n_sources)
    sdr_mixture = np.zeros(n_sources)
    
    for i in range(n_sources):
        # Calculate SDR for the separated source
        sdr_values[i] = _calculate_sdr(estimated_sources[i], target_sources[i])
        
        # Calculate SDR for the mixture
        sdr_mixture[i] = _calculate_sdr(mixture, target_sources[i])
    
    # SDRi = SDR(estimated, target) - SDR(mixture, target)
    sdri_values = sdr_values - sdr_mixture
    
    return sdri_values


def _calculate_sdr(estimated: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Signal-to-Distortion Ratio (SDR) for a single source.
    
    Args:
        estimated: Estimated source signal, shape (n_samples,)
        target: Target source signal, shape (n_samples,)
    
    Returns:
        SDR value in dB
    """
    # Calculate SDR
    numerator = np.sum(target**2)
    denominator = np.sum((target - estimated)**2)
    sdr = 10 * np.log10(numerator / (denominator + 1e-8))
    
    return sdr


def calculate_metrics(
    estimated_sources: Union[np.ndarray, torch.Tensor],
    target_sources: Union[np.ndarray, torch.Tensor],
    mixture: Union[np.ndarray, torch.Tensor],
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate all available metrics for voice separation evaluation.
    
    Args:
        estimated_sources: Estimated source signals, shape (n_sources, n_samples)
                          or single source of shape (n_samples,)
        target_sources: Target source signals, shape (n_sources, n_samples)
                       or single source of shape (n_samples,)
        mixture: Mixture signal, shape (n_samples,)
    
    Returns:
        Dictionary containing all calculated metrics
    """
    # Convert to numpy if tensors
    if isinstance(estimated_sources, torch.Tensor):
        estimated_sources = estimated_sources.detach().cpu().numpy()
    if isinstance(target_sources, torch.Tensor):
        target_sources = target_sources.detach().cpu().numpy()
    if isinstance(mixture, torch.Tensor):
        mixture = mixture.detach().cpu().numpy()
    
    # Handle single source case
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if target_sources.ndim == 1:
        target_sources = target_sources[np.newaxis, :]
    
    # Calculate metrics
    si_snri_values = calculate_si_snri(estimated_sources, target_sources, mixture)
    sdri_values = calculate_sdri(estimated_sources, target_sources, mixture)
    
    # Convert to list for JSON serialization
    if isinstance(si_snri_values, np.ndarray):
        si_snri_values = si_snri_values.tolist()
    if isinstance(sdri_values, np.ndarray):
        sdri_values = sdri_values.tolist()
    
    # Create metrics dictionary
    metrics = {
        "si_snri": si_snri_values,
        "sdri": sdri_values,
        "si_snri_mean": np.mean(si_snri_values),
        "sdri_mean": np.mean(sdri_values),
    }
    
    return metrics