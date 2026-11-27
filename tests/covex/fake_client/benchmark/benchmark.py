"""Fake ASR benchmark client for the bench.sh ordering test: logs each call, sends one POST, writes the schema."""
import argparse, json, os, urllib.request

p = argparse.ArgumentParser()
for a in ["--base-url", "--model", "--data-dir", "--concurrency", "--phases", "--label", "--output"]:
    p.add_argument(a)
for a in ["--num-samples", "--rounds", "--warmup"]:
    p.add_argument(a, type=int)
x = p.parse_args()
with open(os.environ["FAKE_EVENTS"], "a") as f:
    f.write(f"CLIENT c={x.concurrency} ph={x.phases} n={x.num_samples} out={os.path.basename(x.output)}\n")
urllib.request.urlopen(urllib.request.Request(x.base_url + "/v1/audio/transcriptions", data=b"x", method="POST")).read()
json.dump({"phases": {ph: [{"concurrency": int(x.concurrency.split(",")[0]), "audio_seconds_per_s": 1.0}]
                      for ph in x.phases.split(",")}}, open(x.output, "w"))
