# Logbook

## 16.10.23

I have implemented a modular system in pytorch, with the following:
  
<!-- `GammaToneFilter (non-trainable)`: generates a time-domain gammatone filter in the time-domain.  I have added an optional input `impairment_factor` which is simply increasing the ERB of the gammatone filters. -->

1. **FIRFilter1D**:
   - Implements an arbitrary phase FIR (Finite Impulse Response) filter using 1D convolution.
   - The filter taps are initialized as learnable parameters.

2. **FIRFilter1DLinearPhaseI**:
   - Implements a linear phase FIR type I filter using 1D convolution.
   - Only allows an odd number of taps and enforces symmetry in the filter taps.
   - Taps are initialized as learnable parameters.

3. **GammaToneFilter**:
   - Generates a gammatone filter, a type of auditory filter used in audio processing.
   - Impulse response of the filter is computed based on specified parameters like center frequency, sampling frequency, and impairment factor.
   - The filter is applied to the input signal using 1D convolution.

4. **GammaToneFilterbank**:
   - Generates a filterbank of gammatone filters with specified center frequencies.
   - Uses the `GammaToneFilter` module to create individual filters and applies them to the input signal.

5. **MyModel_v1**:
   - A trainable model composed of a normal hearing model and an impaired hearing model.
   - Uses the `NormalModel` and `ImpairedModel` to process the input and produce outputs for normal and impaired hearing, respectively.

6. **ImpairedModel**:
   - Represents an impaired hearing model.
   - Uses a gammatone filterbank and a gain filter to process the input.
   - The gain filter is implemented using `FIRFilter1D`.

7. **NormalModel**:
   - Represents a normal hearing model.
   - Uses a gammatone filterbank to process the input and simulate normal hearing.

I have tried training a `NormalModel` using Stochastic gradent descent with the SI-SDR as the loss function, which success. The results are presented below. 

![img1](link)

## 17.10.23