import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def spectral_loss(target, estimate, n_fft = 1024, hop_length = 256):
    
    batch_size,n_channels,n_samples = estimate.shape

    # reshape audio signal for stft function
    target = target.reshape((batch_size, n_samples))
    estimate = estimate.reshape((batch_size, n_samples))
    
    # compute stfts for each batch
    spec_target = torch.stft(target, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    spec_estimate = torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        
    # convert to dB
    spec_mag_dB = torch.log10(torch.abs(spec_target) + 1e-6)
    spec_estimate_dB = torch.log10(torch.abs(spec_estimate) + 1e-6)
    
    loss = torch.mean(spec_mag_dB-spec_estimate_dB)   
    
    return loss