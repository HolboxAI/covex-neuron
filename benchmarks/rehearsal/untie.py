"""Write an untied copy of a small tied-embedding Qwen3 checkpoint (lm_head = copy of embed_tokens).
vllm-neuron's Qwen3 model has no tied-embedding path (it always loads lm_head.weight, as Qwen3-8B/14B/32B ship it),
so the 0.6B checkpoints are converted to exercise exactly the 32B code path. Usage: untie.py <src_dir> <dst_dir>"""
import json, os, shutil, sys
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1:3]
os.makedirs(dst, exist_ok=True)
shards = [f for f in os.listdir(src) if f.endswith(".safetensors")]
assert len(shards) == 1, f"expected one shard, got {shards}"
for f in os.listdir(src):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), dst)
t = load_file(os.path.join(src, shards[0]))
emb = [k for k in t if k.endswith("embed_tokens.weight")]
assert len(emb) == 1, emb
head = "lm_head.weight"
# Causal-LM checkpoints (model.* names) need lm_head; Qwen3-0.6B already ships one. Embedding checkpoints
# (bare names, like Qwen3-Embedding-8B) ship none and the pooling path needs none: leave them as they are.
if emb[0].startswith("model.") and head not in t:
    t[head] = t[emb[0]].clone()
save_file(t, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
cfg = json.load(open(os.path.join(dst, "config.json")))
cfg["tie_word_embeddings"] = False
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print(f"untied {src} -> {dst}: lm_head present={head in t}, {len(t)} tensors")
