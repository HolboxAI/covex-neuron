"""Device selection and model preparation. Runs inside the GPU / Trainium container.

On Trainium (TorchNeuron, torch-neuronx >= 2.13) prepare() applies what the Neuron
docs and the HF Trainium guide require for good performance:
  * bf16 weights
  * static shapes: tokenizer output padded to a fixed length so each shape compiles once
  * torch.compile(backend="neuron", dynamic=False) with Dynamo automatic dynamic shapes off
On CUDA / CPU it returns the object unchanged, so the same script runs everywhere.
"""

import os
import warnings

import torch

NEURON = "neuron"


def is_neuron():
    """True when TorchNeuron is installed and a Trainium device is visible."""
    accel = getattr(torch, "accelerator", None)
    if accel is not None and hasattr(accel, "current_accelerator"):
        try:
            current = accel.current_accelerator()
            if current is not None:
                return current.type == NEURON
        except RuntimeError:
            pass
    mod = getattr(torch, NEURON, None)
    return bool(mod is not None and mod.is_available())


def device():
    """Return "neuron", "cuda" or "cpu", in that order of preference."""
    if is_neuron():
        return NEURON
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if is_neuron():
    # Transformers' async weight loading is not supported on Neuron (HF Trainium guide).
    # Must be set before from_pretrained, so do it at import time.
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def prepare(obj, tokenizer=None, max_length=128, dtype=torch.bfloat16, compile=True):
    """Make a transformers model, transformers pipeline or diffusers pipeline Trainium-ready.

    Returns the same object (modified in place) so it can wrap an existing assignment:
        model = cn.prepare(model, tokenizer)
        pipe = cn.prepare(pipe)
    """
    if not is_neuron():
        return obj

    import torch._dynamo

    # The Neuron backend cannot compile Dynamo's automatic dynamic shapes yet (TRL #7660).
    torch._dynamo.config.automatic_dynamic_shapes = False

    if _is_diffusers_pipeline(obj):
        _prepare_diffusers(obj, dtype, compile)
        return obj

    model = getattr(obj, "model", obj)  # transformers pipeline or bare model
    tok = tokenizer if tokenizer is not None else getattr(obj, "tokenizer", None)
    if tok is not None:
        _pad_to_fixed_length(tok, max_length)
    model.to(device=NEURON, dtype=dtype)
    if compile:
        model.forward = torch.compile(model.forward, backend=NEURON, dynamic=False)
    return obj


def _is_diffusers_pipeline(obj):
    return hasattr(obj, "components") and any(hasattr(obj, n) for n in ("unet", "transformer"))


def _prepare_diffusers(pipe, dtype, compile):
    pipe.to(NEURON, dtype)
    if not compile:
        return
    for name in ("transformer", "unet"):
        denoiser = getattr(pipe, name, None)
        if denoiser is None:
            continue
        if hasattr(denoiser, "compile_repeated_blocks"):
            # Compiles one transformer block and reuses it: much shorter compile (HF Trainium guide).
            denoiser.compile_repeated_blocks(backend=NEURON, fullgraph=True, dynamic=False)
        else:
            denoiser.forward = torch.compile(denoiser.forward, backend=NEURON, dynamic=False)


def _pad_to_fixed_length(tok, max_length):
    """Force every tokenizer call to pad/truncate to max_length.

    Neuron compiles one graph per input shape; a new sequence length means a new compile
    (~25-140 s each). Patching the class (not the instance) because Python looks up
    __call__ on the type.
    """
    if getattr(tok, "_covex_fixed_length", None) == max_length:
        return
    base = type(tok)

    def __call__(self, *args, **kwargs):
        kwargs["padding"] = "max_length"
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
        return base.__call__(self, *args, **kwargs)

    tok.__class__ = type(base.__name__, (base,), {"__call__": __call__})
    tok._covex_fixed_length = max_length
    if getattr(tok, "pad_token", "") is None:
        warnings.warn("covex_neuron: tokenizer has no pad_token; using eos_token for static padding")
        tok.pad_token = tok.eos_token
