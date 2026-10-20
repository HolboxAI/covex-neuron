"""covex-neuron: run your existing PyTorch / Hugging Face code on AWS Trainium.

In your script (the only change from NVIDIA code):

    import covex_neuron as cn
    device = cn.device()      # "neuron" on Trainium, "cuda" on GPU, else "cpu"
    model = cn.prepare(model) # Trainium: bf16 + static shapes + torch.compile(backend="neuron"); no-op elsewhere

From your laptop or notebook (pay per second, nothing left running):

    cn.run(["sentiment.py"], on="gpu")
    cn.run(["sentiment.py"], on="trainium")
"""

__version__ = "0.0.1.dev1"


def __getattr__(name):
    # torch is only needed inside the job; boto3 only on the laptop. Import each lazily.
    if name in ("device", "is_neuron", "prepare"):
        from covex_neuron import _runtime
        return getattr(_runtime, name)
    if name in ("run", "compare"):
        from covex_neuron import _cloud
        return getattr(_cloud, name)
    raise AttributeError(name)


__all__ = ["device", "is_neuron", "prepare", "run", "compare", "__version__"]
