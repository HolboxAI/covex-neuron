<h1 align="center">covex-neuron</h1>

<p align="center">
    <a href="https://github.com/HolboxAI/covex-neuron/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/HolboxAI/covex-neuron.svg?color=blue"></a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
    <img alt="Status" src="https://img.shields.io/badge/status-pre--alpha-orange">
</p>

<h3 align="center">
    <p>Run your PyTorch and Hugging Face code on AWS Trainium, unchanged from NVIDIA except for two lines</p>
</h3>

covex-neuron is built on [TorchNeuron](https://pytorch.org/blog/introducing-torchneuron-native-pytorch-backend-for-aws-trainium/), the native PyTorch backend for Trainium2 and Trainium3. It handles the parts that make Trainium different from a GPU (bf16, fixed input shapes, the Neuron compiler) and runs your script on AWS by the second, next to an NVIDIA GPU for comparison.

## Installation

```bash
pip install covex-neuron
```

## Quickstart

Add two lines to an existing GPU script:

```python
import covex_neuron as cn

device = cn.device()                      # "neuron" on Trainium, "cuda" on GPU
pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
pipe = cn.prepare(pipe)                   # no-op on GPU / CPU
```

`cn.prepare` takes a transformers model (with its tokenizer), a transformers pipeline, or a diffusers pipeline.

Run it on a GPU and on Trainium from your laptop, using only your AWS credentials:

```python
gpu = cn.run(["sentiment.py"], on="gpu")        # 1x NVIDIA A10G
trn = cn.run(["sentiment.py"], on="trainium")   # 1x Trainium2 chip
cn.compare(gpu, trn)
```

You pay per second while the job runs, and nothing keeps running afterwards.

## Examples

[`examples/nvidia`](examples/nvidia) has sentiment analysis, text classification and image generation scripts as written for a GPU. [`examples/trainium`](examples/trainium) has the same scripts with the two lines added. To run both and compare:

```bash
python examples/run_demo.py
```

## License

[Apache 2.0](LICENSE)
