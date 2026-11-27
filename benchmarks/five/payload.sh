#!/bin/bash
shutdown -h +105
B=covex-neuron-runs-992382417943; R=us-east-2; P=five-model
exec > /var/log/covex-run.log 2>&1
mkdir -p /home/ubuntu/run && chown ubuntu /home/ubuntu/run
( while true; do aws s3 sync /home/ubuntu/run s3://$B/$P/ --region $R --only-show-errors --exclude "profiles/*/*.ntff"; aws s3 cp /var/log/covex-run.log s3://$B/$P/run.log --region $R --only-show-errors; sleep 90; done ) &
sudo -iu ubuntu bash <<'EOS'
set -x
B=covex-neuron-runs-992382417943; R=us-east-2
source "$(ls -d /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0)/bin/activate"
pip freeze > ~/run/pip_freeze.txt
cd ~ && git clone -q -b exp/six-model https://github.com/HolboxAI/covex-neuron && cd covex-neuron && git rev-parse HEAD > ~/run/covex_commit.txt
pip install -q --no-deps -e . && pip install -q httpx soundfile jiwer librosa perfetto huggingface_hub || { echo "SETUP FAILED"; exit 1; }
mkdir -p ~/data ~/models ~/qwen3-asr-inference-cuda-trainium
aws s3 cp s3://$B/inputs/sixdata.tgz - --region $R | tar xz -C ~/data
aws s3 cp s3://$B/inputs/asr_client.tgz - --region $R | tar xz -C ~/qwen3-asr-inference-cuda-trainium
aws s3 cp s3://$B/inputs/asr_data.tgz - --region $R | tar xz -C ~/qwen3-asr-inference-cuda-trainium
ls ~/data/sixdata/vl/manifest.json ~/qwen3-asr-inference-cuda-trainium/data/manifest.json ~/qwen3-asr-inference-cuda-trainium/benchmark/benchmark.py || { echo "DATA FAILED"; exit 1; }
hf download Qwen/Qwen3-32B --local-dir ~/models/Qwen3-32B > /dev/null &
hf download Qwen/Qwen3-Embedding-8B --local-dir ~/models/Qwen3-Embedding-8B > /dev/null &
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir ~/models/Qwen3-VL-32B-Instruct > /dev/null &
hf download openai/gpt-oss-20b --local-dir ~/models/gpt-oss-20b > /dev/null &
hf download Qwen/Qwen3-ASR-1.7B --local-dir ~/models/Qwen3-ASR-1.7B > /dev/null &
wait
for m in Qwen3-32B Qwen3-Embedding-8B Qwen3-VL-32B-Instruct gpt-oss-20b Qwen3-ASR-1.7B; do ls ~/models/$m/config.json || echo "MODEL $m MISSING"; done
du -sh ~/models/*
cd ~/run && bash ~/covex-neuron/benchmarks/five/run.sh
aws s3 sync /var/tmp/neuron-compile-cache s3://$B/compile-cache/five-model/ --region $R --only-show-errors || true
echo ALLDONE
EOS
sleep 120; sudo shutdown -h now
