<h1 align="center">covex-neuron</h1>

<p align="center">
    <a href="https://pypi.org/project/covex-neuron/"><img alt="PyPI" src="https://img.shields.io/pypi/v/covex-neuron?include_prereleases&label=pypi"></a>
    <a href="https://github.com/HolboxAI/covex-neuron"><img alt="GitHub" src="https://img.shields.io/badge/github-HolboxAI%2Fcovex--neuron-black?logo=github"></a>
    <a href="https://github.com/HolboxAI/covex-neuron/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/HolboxAI/covex-neuron.svg?color=blue"></a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
    <img alt="Status" src="https://img.shields.io/badge/status-pre--alpha-orange">
</p>

<h3 align="center">
    <p>Run your PyTorch and Hugging Face code on AWS Trainium, unchanged from NVIDIA except for two lines</p>
</h3>

<p align="center">
    <a href="https://github.com/HolboxAI/covex-neuron">GitHub</a> ·
    <a href="https://pypi.org/project/covex-neuron/">PyPI</a> ·
    <a href="https://github.com/HolboxAI/covex-neuron/tree/main/notebooks">Notebooks</a>
</p>

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
gpu = cn.run(["sentiment.py"], on="gpu")        # 1x NVIDIA GPU, g5.xlarge Spot
trn = cn.run(["sentiment.py"], on="trainium")   # 1x Trainium2 chip, trn2.3xlarge Spot
cn.compare(gpu, trn)
```

Both run as EC2 Spot instances in sa-east-1 on AWS's Deep Learning AMIs, are billed per second, and terminate themselves when the script finishes. Your account needs EC2 Spot quota there for G instances and Trn instances.

## Notebooks

| Notebook | Workload |
|---|---|
| [01 Sentiment analysis](https://github.com/HolboxAI/covex-neuron/blob/main/notebooks/01_sentiment_analysis.ipynb) | transformers pipeline, DistilBERT SST-2 |
| [02 Text classification](https://github.com/HolboxAI/covex-neuron/blob/main/notebooks/02_text_classification.ipynb) | AutoModelForSequenceClassification, emotion (7 classes) |
| [03 Image generation](https://github.com/HolboxAI/covex-neuron/blob/main/notebooks/03_image_generation.ipynb) | diffusers, SD-Turbo |

Each one runs the same script on a GPU and on Trainium and prints the comparison.

## Examples

[`examples/nvidia`](https://github.com/HolboxAI/covex-neuron/blob/main/examples/nvidia) has sentiment analysis, text classification and image generation scripts as written for a GPU. [`examples/trainium`](https://github.com/HolboxAI/covex-neuron/blob/main/examples/trainium) has the same scripts with the two lines added. To run both and compare:

```bash
python examples/run_demo.py
```

## License

[Apache 2.0](https://github.com/HolboxAI/covex-neuron/blob/main/LICENSE)
