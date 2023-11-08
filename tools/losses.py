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

    loss = torch.norm(log_stft_target - log_stft_estimate, p='fro')

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

    loss = torch.norm(abs_stft_target - abs_stft_estimate, p='fro')

    return loss


# def weighted_phase_loss(target,estimate,n_fft,overlap):
#     batch_size, _, n_samples = estimate.shape

#     # reshape audio signal for stft function
#     target = target.reshape((batch_size, n_samples))
#     estimate = estimate.reshape((batch_size, n_samples))

#     hop_length = int((1 - overlap) * n_fft)

#     # compute stfts for each batch
#     stft_target = torch.stft(
#         target, n_fft=n_fft, hop_length=hop_length, return_complex=True
#     )
#     stft_estimate = torch.stft(
#         estimate, n_fft=n_fft, hop_length=hop_length, return_complex=True
#     )

#     phase_target = torch.angle(stft_target)
#     abs_target = torch.abs(stft_target)
#     phase_estimate = torch.angle(stft_estimate)
#     abs_estimate = torch.abs(stft_estimate)

#     loss = abs_target * abs_estimate
#     loss -= torch.real(stft_target) * torch.real(stft_estimate)
#     loss -= torch.imag(stft_target) * torch.imag(stft_estimate)
#     return torch.norm(loss, p=1)


# def combined_loss(target,estimate,n_fft,overlap):
#     a = l2_stft_loss(target,estimate,n_fft,overlap)
#     b = l1_stft_loss(target,estimate,n_fft,overlap)
#     c = log_l1_stft_loss(target,estimate,n_fft,overlap)
#     return a + b + c
