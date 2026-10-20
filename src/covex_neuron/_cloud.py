"""Run scripts on an NVIDIA GPU or AWS Trainium, pay per second, nothing left running.

    gpu       SageMaker training job on ml.g5.xlarge (1x A10G), AWS PyTorch GPU container.
    trainium  1-chip ml.trn2.3xlarge, the smallest Trainium2. AWS only offers it through SageMaker
              HyperPod, so we create a one-node cluster, run, upload results and delete the cluster.
              The node also deletes its own cluster when done, so nothing runs on if the laptop dies.

The caller needs only AWS credentials. The S3 bucket and IAM role are created on first use.
"""

import io
import json
import os
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

DEFAULT_REGION = "us-west-2"  # the only region with ml.trn2.3xlarge on SageMaker HyperPod (Oct 2026)
ROLE_NAME = "CovexNeuronRole"
TAG = {"Key": "covex-neuron", "Value": "true"}
DEFAULT_REQUIREMENTS = ["transformers>=5", "diffusers", "accelerate", "pillow"]

GPU_INSTANCE = "ml.g5.xlarge"
GPU_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker"
TRN_INSTANCE = "ml.trn2.3xlarge"
# Neuron SDK 2.33 container with TorchNeuron (torch-neuronx 2.14, Python 3.14). Pulled by Docker on the node.
TRN_IMAGE = "public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04"
MIN_NEURON_DRIVER = (2, 31)  # needed by the 2.33 runtime (Neuron 2.33 release notes)

RESULTS_DIR = Path("covex-results")


def run(scripts, on, args=(), checkpoint=None, region=DEFAULT_REGION, requirements=None, max_minutes=45):
    """Run one or more scripts (same folder) back to back on one machine; return the local results dir.

    Each script's outputs ($SM_MODEL_DIR) land in covex-results/<job>/<script name>/.
    `checkpoint` (local folder or s3:// prefix) is mounted and passed to every script as --model <path>.
    """
    scripts = [scripts] if isinstance(scripts, (str, Path)) else list(scripts)
    folders = {Path(s).resolve().parent for s in scripts}
    if len(folders) != 1:
        raise ValueError("all scripts must be in the same folder (it is uploaded as the job's code)")
    if on not in ("gpu", "trainium"):
        raise ValueError('on must be "gpu" or "trainium"')

    aws = _Aws(region)
    job = f"covex-{'gpu' if on == 'gpu' else 'trn'}-{time.strftime('%Y%m%d-%H%M%S')}"
    prefix = f"jobs/{job}"
    aws.s3.put_object(Bucket=aws.bucket, Key=f"{prefix}/code/code.tar.gz", Body=_bundle(folders.pop()))

    args = list(args)
    model_s3 = None
    if checkpoint:
        model_s3 = checkpoint if str(checkpoint).startswith("s3://") else aws.upload_dir(checkpoint)
    reqs = requirements or DEFAULT_REQUIREMENTS
    names = [Path(s).name for s in scripts]

    if on == "gpu":
        out = _run_training_job(aws, job, prefix, names, args, model_s3, reqs, max_minutes)
    else:
        out = _run_hyperpod(aws, job, prefix, names, args, model_s3, reqs, max_minutes)
    print(f"results: {out}")
    return out


# ---------------------------------------------------------------- shared

class _Aws:
    def __init__(self, region):
        self.region = region
        sess = boto3.session.Session(region_name=region)
        self.s3 = sess.client("s3")
        self.sm = sess.client("sagemaker")
        self.logs = sess.client("logs")
        self.iam = sess.client("iam")
        self.account = sess.client("sts").get_caller_identity()["Account"]
        # HyperPod's instance policy only allows buckets named sagemaker-*.
        self.bucket = f"sagemaker-covex-neuron-{self.account}-{region}"
        self._ensure_bucket()
        self.role_arn = self._ensure_role()

    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            kwargs = {} if self.region == "us-east-1" else {
                "CreateBucketConfiguration": {"LocationConstraint": self.region}}
            self.s3.create_bucket(Bucket=self.bucket, **kwargs)
            print(f"created s3://{self.bucket}")

    def _ensure_role(self):
        try:
            return self.iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        except self.iam.exceptions.NoSuchEntityException:
            pass
        trust = {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        arn = self.iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        for policy in ("AmazonSageMakerFullAccess", "AmazonSageMakerClusterInstanceRolePolicy"):
            self.iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=f"arn:aws:iam::aws:policy/{policy}")
        # Lets a HyperPod node delete its own (tagged) cluster when the run is over.
        self.iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="covex-self-delete", PolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Action": "sagemaker:DeleteCluster", "Resource": "*",
                "Condition": {"StringEquals": {"aws:ResourceTag/covex-neuron": "true"}}}]}))
        print(f"created IAM role {arn}; waiting for it to propagate")
        time.sleep(15)
        return arn

    def upload_dir(self, local):
        local = Path(local).resolve()
        prefix = f"checkpoints/{local.name}"
        for f in local.rglob("*"):
            if f.is_file():
                self.s3.upload_file(str(f), self.bucket, f"{prefix}/{f.relative_to(local)}")
        return f"s3://{self.bucket}/{prefix}"

    def download_results(self, key, job):
        out = RESULTS_DIR / job
        out.mkdir(parents=True, exist_ok=True)
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError:
            print("no outputs were uploaded")
            return out
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
            tar.extractall(out)
        return out

    def price_per_hour(self, instance, component):
        pricing = boto3.client("pricing", region_name="us-east-1")
        resp = pricing.get_products(ServiceCode="AmazonSageMaker", Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceName", "Value": instance},
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": self.region},
            {"Type": "TERM_MATCH", "Field": "component", "Value": component},
        ])
        for item in resp["PriceList"]:
            for term in json.loads(item)["terms"].get("OnDemand", {}).values():
                for dim in term["priceDimensions"].values():
                    return float(dim["pricePerUnit"]["USD"])
        return None


def _bundle(folder):
    """tar.gz of the scripts' folder plus this package's source (put on PYTHONPATH in the job)."""
    def skip(info):
        name = os.path.basename(info.name)
        return None if name in ("__pycache__", ".git", ".venv", "covex-results") or name.endswith(".pyc") else info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(folder, arcname=".", filter=skip)
        tar.add(Path(__file__).resolve().parent, arcname="_covex_src/covex_neuron", filter=skip)
    return buf.getvalue()


def _job_command(code_dir, out_dir, model_dir, names, args, reqs):
    """Shell run inside the container: install requirements, then each script with its own output dir."""
    q = lambda s: "'" + str(s).replace("'", "'\"'\"'") + "'"
    script_args = " ".join(q(a) for a in args) + (f" --model {model_dir}" if model_dir else "")
    lines = [
        "set -uo pipefail",
        f"mkdir -p /tmp/code && tar xzf {code_dir}/code.tar.gz -C /tmp/code && cd /tmp/code",
        "export PYTHONPATH=/tmp/code/_covex_src:${PYTHONPATH:-} HF_HOME=/tmp/hf PYTHONUNBUFFERED=1",
        "pip install -q " + " ".join(q(r) for r in reqs),
        "python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__)'",
        "status=0",
    ]
    for name in names:
        stem = Path(name).stem
        lines += [
            f"echo '===== {name} ====='",
            f"SM_MODEL_DIR={out_dir}/{stem} python -u {name} {script_args} || {{ echo 'FAILED: {name}'; status=1; }}",
        ]
    lines.append("exit $status")
    return "\n".join(lines)


def _write_job_meta(out, **meta):
    (out / "job.json").write_text(json.dumps(meta, indent=2, default=str))


# ---------------------------------------------------------------- GPU: SageMaker training job

def _run_training_job(aws, job, prefix, names, args, model_s3, reqs, max_minutes):
    channel = lambda name, uri: {"ChannelName": name, "DataSource": {"S3DataSource": {
        "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}}}
    inputs = [channel("code", f"s3://{aws.bucket}/{prefix}/code/")]
    if model_s3:
        inputs.append(channel("model", model_s3))
    command = _job_command("/opt/ml/input/data/code", "/opt/ml/model",
                           "/opt/ml/input/data/model" if model_s3 else None, names, args, reqs)
    aws.sm.create_training_job(
        TrainingJobName=job,
        RoleArn=aws.role_arn,
        AlgorithmSpecification={"TrainingImage": GPU_IMAGE.format(region=aws.region), "TrainingInputMode": "File",
                                "ContainerEntrypoint": ["bash", "-c"], "ContainerArguments": [command]},
        InputDataConfig=inputs,
        OutputDataConfig={"S3OutputPath": f"s3://{aws.bucket}/{prefix}/output"},
        ResourceConfig={"InstanceType": GPU_INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 100},
        StoppingCondition={"MaxRuntimeInSeconds": max_minutes * 60},
        Tags=[TAG],
    )
    print(f"started {job} on {GPU_INSTANCE}")

    group, tokens = "/aws/sagemaker/TrainingJobs", {}
    while True:
        desc = aws.sm.describe_training_job(TrainingJobName=job)
        _print_cloudwatch(aws.logs, group, job, tokens)
        if desc["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(15)
    _print_cloudwatch(aws.logs, group, job, tokens)
    status = desc["TrainingJobStatus"]
    print(f"{job}: {status}" + (f" - {desc.get('FailureReason')}" if status == "Failed" else ""))

    out = aws.download_results(f"{prefix}/output/{job}/output/model.tar.gz", job)
    price = aws.price_per_hour(GPU_INSTANCE, "Training")
    billable = desc.get("BillableTimeInSeconds", 0)
    _write_job_meta(out, job=job, target="gpu", instance=GPU_INSTANCE, status=status, billable_seconds=billable,
                    price_per_hour=price, cost_usd=price * billable / 3600 if price else None)
    return out


def _print_cloudwatch(logs, group, prefix, tokens):
    try:
        streams = logs.describe_log_streams(logGroupName=group, logStreamNamePrefix=prefix)["logStreams"]
    except logs.exceptions.ResourceNotFoundException:
        return
    for stream in streams:
        name = stream["logStreamName"]
        kwargs = {"logGroupName": group, "logStreamName": name, "startFromHead": True}
        if name in tokens:
            kwargs["nextToken"] = tokens[name]
        resp = logs.get_log_events(**kwargs)
        for event in resp["events"]:
            print(event["message"])
        tokens[name] = resp["nextForwardToken"]


# ---------------------------------------------------------------- Trainium: one-node HyperPod cluster

def _run_hyperpod(aws, job, prefix, names, args, model_s3, reqs, max_minutes):
    s3_root = f"s3://{aws.bucket}/{prefix}"
    command = _job_command("/code", "/out", "/model" if model_s3 else None, names, args, reqs)
    aws.s3.put_object(Bucket=aws.bucket, Key=f"{prefix}/lifecycle/on_create.sh", Body=_ON_CREATE.encode())
    aws.s3.put_object(Bucket=aws.bucket, Key=f"{prefix}/lifecycle/covex_job.sh", Body=_node_script(
        aws.region, s3_root, model_s3, job, command, max_minutes).encode())

    aws.sm.create_cluster(
        ClusterName=job,
        InstanceGroups=[{
            "InstanceGroupName": "trn",
            "InstanceType": TRN_INSTANCE,
            "InstanceCount": 1,
            "LifeCycleConfig": {"SourceS3Uri": f"{s3_root}/lifecycle/", "OnCreate": "on_create.sh"},
            "ExecutionRole": aws.role_arn,
        }],
        NodeRecovery="None",
        Tags=[TAG],
    )
    started = datetime.now(timezone.utc)
    print(f"creating 1-node cluster {job} ({TRN_INSTANCE}); provisioning usually takes 5-15 min")

    log_key, printed, deadline = f"{prefix}/log.txt", 0, time.time() + (max_minutes + 30) * 60
    status, done = None, False
    try:
        while time.time() < deadline:
            try:
                status = aws.sm.describe_cluster(ClusterName=job)["ClusterStatus"]
            except ClientError:
                status = "Deleted"
            try:
                text = aws.s3.get_object(Bucket=aws.bucket, Key=log_key)["Body"].read().decode(errors="replace")
                print(text[printed:], end="")
                printed = len(text)
            except ClientError:
                pass
            if _exists(aws, f"{prefix}/DONE"):
                done = True
                break
            if status == "Failed":
                print(f"cluster failed: {aws.sm.describe_cluster(ClusterName=job).get('FailureMessage')}")
                break
            time.sleep(20)
    finally:
        _delete_cluster(aws, job)
    minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60

    out = aws.download_results(f"{prefix}/output/out.tar.gz", job)
    price = aws.price_per_hour(TRN_INSTANCE, "Cluster")
    _write_job_meta(out, job=job, target="trainium", instance=TRN_INSTANCE, status="Completed" if done else status,
                    approx_billed_minutes=round(minutes, 1), price_per_hour=price,
                    cost_usd=price * minutes / 60 if price else None,
                    note="HyperPod bills while the node exists; minutes counted from create to delete request")
    return out


def _exists(aws, key):
    try:
        aws.s3.head_object(Bucket=aws.bucket, Key=key)
        return True
    except ClientError:
        return False


def _delete_cluster(aws, name):
    try:
        aws.sm.delete_cluster(ClusterName=name)
        print(f"deleting cluster {name}")
    except ClientError as e:
        if not any(s in str(e) for s in ("ResourceNotFound", "does not exist", "Deleting")):
            print(f"warning: could not delete cluster {name}: {e} -- delete it in the console to stop billing")


# HyperPod waits on OnCreate, so it only starts the real work in the background and returns.
_ON_CREATE = """#!/bin/bash
cp covex_job.sh /tmp/covex_job.sh
nohup bash /tmp/covex_job.sh > /var/log/covex_job.log 2>&1 &
exit 0
"""


def _node_script(region, s3_root, model_s3, cluster, command, max_minutes):
    major, minor = MIN_NEURON_DRIVER
    model_mount = "-v /tmp/model:/model" if model_s3 else ""
    model_sync = f"aws s3 sync --quiet {model_s3} /tmp/model" if model_s3 else ""
    return f"""#!/bin/bash
set -u
export AWS_DEFAULT_REGION={region}
LOG=/var/log/covex_job.log
command -v aws >/dev/null || pip3 install -q awscli
( while true; do aws s3 cp --quiet $LOG {s3_root}/log.txt; sleep 15; done ) &
UPLOADER=$!
finish() {{
  tar czf /tmp/out.tar.gz -C /tmp/out . 2>/dev/null && aws s3 cp --quiet /tmp/out.tar.gz {s3_root}/output/out.tar.gz
  kill $UPLOADER; aws s3 cp --quiet $LOG {s3_root}/log.txt
  echo done | aws s3 cp --quiet - {s3_root}/DONE
  aws sagemaker delete-cluster --cluster-name {cluster}   # stop billing even if the laptop is gone
}}
trap finish EXIT
mkdir -p /tmp/out /tmp/code /tmp/model
echo "[covex] node up: $(hostname), $(ls /dev/neuron* 2>/dev/null | tr '\\n' ' ')"

if ! command -v docker >/dev/null; then
  echo "[covex] installing docker"; apt-get update -qq && apt-get install -y -qq docker.io && systemctl start docker
fi

drv=$(modinfo -F version neuron 2>/dev/null || echo 0.0)
echo "[covex] neuron driver $drv"
if [ "$(printf '%s\\n' "$drv" "{major}.{minor}" | sort -V | head -1)" != "{major}.{minor}" ]; then
  echo "[covex] upgrading neuron driver (need >= {major}.{minor})"
  . /etc/os-release
  curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --dearmor -o /usr/share/keyrings/neuron.gpg
  echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com $VERSION_CODENAME main" > /etc/apt/sources.list.d/neuron.list
  apt-get update -qq && apt-get install -y -qq aws-neuronx-dkms && (rmmod neuron; modprobe neuron)
  echo "[covex] neuron driver now $(modinfo -F version neuron)"
fi

aws s3 cp --quiet {s3_root}/code/code.tar.gz /tmp/code/code.tar.gz
{model_sync}
echo "[covex] pulling {TRN_IMAGE}"
docker pull -q {TRN_IMAGE}
DEVICES=$(for d in /dev/neuron*; do printf -- '--device=%s ' "$d"; done)
timeout {max_minutes}m docker run --rm $DEVICES -v /tmp/code:/code -v /tmp/out:/out {model_mount} \\
  {TRN_IMAGE} bash -c {_shell_quote(command)}
echo "[covex] exit code $?"
"""


def _shell_quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---------------------------------------------------------------- comparison

def compare(*result_dirs):
    """Print a GPU vs Trainium table per script from covex-results/<job>/ folders."""
    jobs = [Path(d) for d in result_dirs]
    metas = [json.loads((d / "job.json").read_text()) for d in jobs]
    scripts = sorted({p.parent.name for d in jobs for p in d.glob("*/metrics.json")})
    head = ["metric"] + [f"{m['target']} ({m['instance']})" for m in metas]
    for script in scripts:
        print(f"\n== {script}")
        rows = [head]
        metrics = [_read(d / script / "metrics.json", {}) for d in jobs]
        for key, label in [("device", "device"), ("load_s", "model load (s)"),
                           ("first_batch_s", "first batch incl. compile (s)"),
                           ("warm_latency_ms_p50", "warm latency p50 (ms)"),
                           ("warm_latency_ms_p95", "warm latency p95 (ms)"),
                           ("throughput_items_per_s", "throughput (items/s)")]:
            rows.append([label] + [_fmt(m.get(key)) for m in metrics])
        preds = [_read(d / script / "predictions.json", None) for d in jobs]
        if len(preds) == 2 and all(preds):
            same = sum(a == b for a, b in zip(*preds))
            rows.append(["same prediction as first column", "-", f"{same}/{min(map(len, preds))}"])
        images = [sorted(str(p) for p in (d / script).glob("*.png")) for d in jobs]
        if any(images):
            rows.append(["images"] + [str(len(i)) + " in " + str(d / script) for i, d in zip(images, jobs)])
        _print_table(rows)
    print("\n== cost of the whole run")
    _print_table([head, ["cost ($)"] + [_fmt(m.get("cost_usd")) for m in metas],
                  ["$/hour"] + [_fmt(m.get("price_per_hour")) for m in metas]])


def _read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if v < 10 else f"{v:,.1f}"
    return "-" if v is None else str(v)


def _print_table(rows):
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
