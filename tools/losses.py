import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def log_l2_stft_loss(target, estimate, n_fft, overlap):
    batch_size, _, n_samples = estimate.shape

    # reshape audio signal for stft function
    target = target.reshape((batch_size, n_samples))
    estimate = estimate.reshape((batch_size, n_samples))

    hop_length = int((1 - overlap) * n_fft)

    # compute stfts for each batch
    stft_target = torch.stft(
        target, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )
    stft_estimate = torch.stft(
        estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )

    log_stft_target = torch.log10(torch.abs(stft_target) + 1e-6)
    log_stft_estimate = torch.log10(torch.abs(stft_estimate) + 1e-6)

    loss = torch.norm(log_stft_target - log_stft_estimate, p="fro")

    return loss


def log_l1_stft_loss(target, estimate, n_fft, overlap):
    batch_size, _, n_samples = estimate.shape

    # reshape audio signal for stft function
    target = target.reshape((batch_size, n_samples))
    estimate = estimate.reshape((batch_size, n_samples))

    hop_length = int((1 - overlap) * n_fft)

    # compute stfts for each batch
    stft_target = torch.stft(
        target, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )
    stft_estimate = torch.stft(
        estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )

    log_stft_target = torch.log10(torch.abs(stft_target) + 1e-6)
    log_stft_estimate = torch.log10(torch.abs(stft_estimate) + 1e-6)

    loss = torch.norm(log_stft_target - log_stft_estimate, p=1)

    return loss


def l1_stft_loss(target, estimate, n_fft, overlap):
    """https://static1.squarespace.com/static/5554d97de4b0ee3b50a3ad52/t/5fb1e9031c7089551a30c2e4/1605495044128/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf"""
    batch_size, _, n_samples = estimate.shape

    # reshape audio signal for stft function
    target = target.reshape((batch_size, n_samples))
    estimate = estimate.reshape((batch_size, n_samples))

    hop_length = int((1 - overlap) * n_fft)

    # compute stfts for each batch
    stft_target = torch.stft(
        target, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )
    stft_estimate = torch.stft(
        estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )

    abs_stft_target = torch.abs(stft_target)
    abs_stft_estimate = torch.abs(stft_estimate)

    loss = torch.norm(abs_stft_target - abs_stft_estimate, p=1)

    return loss


def l2_stft_loss(target, estimate, n_fft, overlap):
    """https://static1.squarespace.com/static/5554d97de4b0ee3b50a3ad52/t/5fb1e9031c7089551a30c2e4/1605495044128/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf"""
    batch_size, _, n_samples = estimate.shape

    # reshape audio signal for stft function
    target = target.reshape((batch_size, n_samples))
    estimate = estimate.reshape((batch_size, n_samples))

    hop_length = int((1 - overlap) * n_fft)

    # compute stfts for each batch
    stft_target = torch.stft(
        target, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )
    stft_estimate = torch.stft(
        estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True
    )

    abs_stft_target = torch.abs(stft_target)
    abs_stft_estimate = torch.abs(stft_estimate)

    loss = torch.norm(abs_stft_target - abs_stft_estimate, p="fro")

    return loss


def multi_scale_STFT_loss(
    target,
    estimate,
    fft_sizes=(16, 32, 64, 128, 256, 512),
    norm="frobenius",
    mag_weight=1.0,
    log_weight=0.0,
    cos_weight=0.0,
    overlap=0.5,
):
    # init loss
    loss = 0
    batch_size, _, n_samples = estimate.shape

    # reshape audio signal for stft function
    target = target.reshape((batch_size, n_samples))
    estimate = estimate.reshape((batch_size, n_samples))

    for n_fft in fft_sizes:
        hop_length = int((1 - overlap) * n_fft)

        # compute stfts for each batch
        stft_target = torch.stft(
            target, n_fft=n_fft, hop_length=hop_length, return_complex=True
        )
        stft_estimate = torch.stft(
            estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True
        )
        abs_stft_target = torch.abs(stft_target)
        abs_stft_estimate = torch.abs(stft_estimate)

        if norm == "frobenius" or "l2":
            loss += torch.norm(abs_stft_target - abs_stft_estimate, p="fro")
        if norm == "l1":
            loss += torch.norm(abs_stft_target - abs_stft_estimate, p=1)

    return loss

def spectral_loss(
    target_audio,
    estimate_audio,
    fft_sizes=(2048, 1024, 512, 256, 128, 64),
    loss_type="L1",
    overlap=0.5,
    mag_weight=1.0,
    delta_time_weight=0.0,
    delta_freq_weight=0.0,
    cumsum_freq_weight=0.0,
    logmag_weight=0.0,
    loudness_weight=0.0,
    weights=None,
):
    """Multi-scale spectral loss adapted from https://github.com/magenta/ddsp/blob/main/ddsp/losses.py

    Args:
        target_audio (torch.tensor): audio target
        estimate_audio (torch.tensor): audio estimate
        fft_sizes (tuple, optional): fft sizes for multi-scale spectrogram comparison. Defaults to (2048, 1024, 512, 256, 128, 64).
        loss_type (str, optional): Can be "L1", "L2" or "COSINE". Defaults to "L1".
        overlap (float, optional): Overlap for spectrogram computation. Defaults to 0.5.
        mag_weight (float, optional): _description_. Defaults to 1.0.
        delta_time_weight (float, optional): _description_. Defaults to 0.0.
        delta_freq_weight (float, optional): _description_. Defaults to 0.0.
        cumsum_freq_weight (float, optional): _description_. Defaults to 0.0.
        logmag_weight (float, optional): _description_. Defaults to 0.0.
        loudness_weight (float, optional): _description_. Defaults to 0.0. TODO make loudness work
        weights (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.tensor: Loss tensor
    """
    loss = 0.0
    batch_size, _, n_samples = estimate_audio.shape
        
    # reshape audio signal for stft function
    target_audio = target_audio.reshape((batch_size, n_samples))
    estimate_audio = estimate_audio.reshape((batch_size, n_samples))

    for fft_size in fft_sizes:
        hop_length = int((1 - overlap) * fft_size)
        target_mag = torch.abs(
            torch.stft(target_audio, fft_size, hop_length, return_complex=True)
        )
        estimate_mag = torch.abs(
            torch.stft(estimate_audio, fft_size, hop_length, return_complex=True)
        )

        if mag_weight > 0:
            loss += mag_weight * mean_difference(
                target_mag, estimate_mag, loss_type, weights
            )

        if delta_time_weight > 0:
            target = torch.diff(target_mag, dim=2)
            value = torch.diff(estimate_mag, dim=2)
            loss += delta_time_weight * mean_difference(
                target, value, loss_type, weights
            )

        if delta_freq_weight > 0:
            target = torch.diff(target_mag, dim=1)
            value = torch.diff(estimate_mag, dim=1)
            loss += delta_freq_weight * mean_difference(
                target, value, loss_type, weights
            )

        if cumsum_freq_weight > 0:
            target = torch.cumsum(target_mag, dim=1)
            value = torch.cumsum(estimate_mag, dim=1)
            loss += cumsum_freq_weight * mean_difference(
                target, value, loss_type, weights
            )

        if logmag_weight > 0:
            target = torch.log(target_mag + 1e-6)
            value = torch.log(estimate_mag + 1e-6)
            loss += logmag_weight * mean_difference(target, value, loss_type, weights)
    try:
        if loudness_weight > 0:
            target = compute_loudness(target, n_fft=2048)
            value = compute_loudness(estimate_audio, n_fft=2048)
            loss += loudness_weight * mean_difference(target, value, loss_type, weights)
    except: 
        pass

    return loss


def mean_difference(target, value, loss_type, weights):
    if loss_type == "L1":
        loss = torch.abs(target - value)
    elif loss_type == "L2":
        loss = (target - value).pow(2)
    elif loss_type == "COSINE":
        target_norm = target.norm(dim=-1, keepdim=True)
        value_norm = value.norm(dim=-1, keepdim=True)
        similarity = torch.sum(target * value, dim=-1) / (target_norm * value_norm)
        loss = 1 - similarity
    else:
        raise ValueError("Invalid loss_type. Use 'L1', 'L2', or 'COSINE'.")

    if weights is not None:
        loss = loss * weights

    return loss.mean()


def compute_loudness(audio, n_fft):
    raise NotImplementedError
