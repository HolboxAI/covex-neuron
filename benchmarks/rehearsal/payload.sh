#!/bin/bash
# x86 CPU rehearsal payload (m7i.16xlarge, Neuron DLAMI). Hard stop 65 min; the instance terminates on shutdown.
shutdown -h +65
B=covex-neuron-runs-992382417943; R=us-east-2; P=rehearsal
exec > /var/log/covex-run.log 2>&1
mkdir -p /home/ubuntu/run && chown ubuntu /home/ubuntu/run
( while true; do aws s3 sync /home/ubuntu/run s3://$B/$P/ --region $R --only-show-errors --exclude "neff/*"; aws s3 cp /var/log/covex-run.log s3://$B/$P/run.log --region $R --only-show-errors; free -g | sed -n 2p >> /home/ubuntu/run/mem.log; sleep 60; done ) &
sudo -iu ubuntu bash <<'EOS'
set -x
B=covex-neuron-runs-992382417943; R=us-east-2
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
pip freeze > ~/run/pip_freeze.txt
cd ~ && git clone -q -b exp/six-model https://github.com/HolboxAI/covex-neuron && cd covex-neuron && git rev-parse HEAD > ~/run/covex_commit.txt
pip install -q --no-deps -e . && pip install -q httpx soundfile jiwer librosa huggingface_hub || { echo "SETUP FAILED"; exit 1; }
mkdir -p ~/data ~/models ~/models-small ~/hf-small ~/qwen3-asr-inference-cuda-trainium
aws s3 cp s3://$B/inputs/sixdata.tgz - --region $R | tar xz -C ~/data
aws s3 cp s3://$B/inputs/asr_client.tgz - --region $R | tar xz -C ~/qwen3-asr-inference-cuda-trainium
aws s3 cp s3://$B/inputs/asr_data.tgz - --region $R | tar xz -C ~/qwen3-asr-inference-cuda-trainium
ls ~/data/sixdata/vl/manifest.json ~/qwen3-asr-inference-cuda-trainium/data/manifest.json || { echo "DATA FAILED"; exit 1; }
# Small same-architecture checkpoints for CPU mode (Qwen3 dense ones untied to match the 32B/8B code path).
hf download Qwen/Qwen3-0.6B --local-dir ~/hf-small/Qwen3-0.6B > /dev/null &
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ~/hf-small/Qwen3-Embedding-0.6B > /dev/null &
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ~/models-small/Qwen3-VL-32B-Instruct > /dev/null &
hf download Qwen/Qwen3-ASR-1.7B --local-dir ~/models/Qwen3-ASR-1.7B > /dev/null && touch ~/models/Qwen3-ASR-1.7B/.done &
wait
python3 ~/covex-neuron/benchmarks/rehearsal/untie.py ~/hf-small/Qwen3-0.6B ~/models-small/Qwen3-32B || echo "UNTIE FAILED q32"
python3 ~/covex-neuron/benchmarks/rehearsal/untie.py ~/hf-small/Qwen3-Embedding-0.6B ~/models-small/Qwen3-Embedding-8B || echo "UNTIE FAILED emb"
ln -sfn ~/models/Qwen3-ASR-1.7B ~/models-small/Qwen3-ASR-1.7B
for m in Qwen3-32B Qwen3-Embedding-8B Qwen3-VL-32B-Instruct Qwen3-ASR-1.7B; do ls ~/models-small/$m/config.json || echo "SMALL $m MISSING"; done
# Real checkpoints for CPU compile, in the background; rehearse.sh waits on each .done marker.
for m in Qwen3-32B Qwen3-VL-32B-Instruct Qwen3-Embedding-8B; do
  ( hf download Qwen/$m --local-dir ~/models/$m > /dev/null && ls ~/models/$m/config.json && touch ~/models/$m/.done ) &
done
cd ~/run && bash ~/covex-neuron/benchmarks/rehearsal/rehearse.sh
mkdir -p ~/run/neff && cp -r ~/neff_cache ~/run/neff/ 2>/dev/null; aws s3 sync ~/neff_cache s3://$B/compile-cache/rehearsal/ --region $R --only-show-errors || true
echo ALLDONE
EOS
sleep 90; sudo shutdown -h now
