"""Same three HF workloads on an NVIDIA GPU and on AWS Trainium, pay per second.

    python examples/run_demo.py

GPU runs the untouched NVIDIA scripts (examples/nvidia). Trainium runs the same scripts with the
two covex lines added (examples/trainium; `diff -r examples/nvidia examples/trainium` shows them).
"""

from pathlib import Path

import covex_neuron as cn

HERE = Path(__file__).parent
SCRIPTS = ["sentiment.py", "text_classify.py", "image_gen.py"]

gpu = cn.run([HERE / "nvidia" / s for s in SCRIPTS], on="gpu")
trn = cn.run([HERE / "trainium" / s for s in SCRIPTS], on="trainium")
cn.compare(gpu, trn)
