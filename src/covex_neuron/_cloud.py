"""Run scripts on an NVIDIA GPU or AWS Trainium, pay per second, nothing left running.

    gpu                EC2 Spot, 1 small NVIDIA GPU (g5.xlarge A10G, else g6.xlarge L4, else g4dn.xlarge T4)
                       with AWS's GPU PyTorch Deep Learning AMI.
    trainium           EC2 Spot trn2.3xlarge (1 Trainium2 chip) with AWS's Neuron DLAMI, which has
                       TorchNeuron preinstalled. sa-east-1 is the only default-enabled region selling it on EC2.
    Both Spot targets run in sa-east-1, upload results to S3 and terminate themselves.

    gpu-sagemaker      On-demand SageMaker training job on ml.g5.xlarge, AWS PyTorch GPU container.
    trainium-hyperpod  On-demand 1-chip ml.trn2.3xlarge via a one-node SageMaker HyperPod cluster
                       (us-west-2), created and deleted around the run. Slower to start, not interruptible.

The caller needs only AWS credentials. S3 buckets and IAM roles are created on first use.
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

SPOT_REGION = "sa-east-1"
EC2_ROLE_NAME = "CovexNeuronEC2Role"
SPOT_TARGETS = {
    "gpu": {
        "instance_types": ["g5.xlarge", "g6.xlarge", "g4dn.xlarge"],
        "ami": "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04) *",
        "venv_check": "import torch; assert torch.cuda.is_available()",
    },
    "trainium": {
        "instance_types": ["trn2.3xlarge"],
        # TorchNeuron (torch-neuronx 2.14, Neuron SDK 2.33) preinstalled.
        "ami": "Deep Learning AMI Neuron torch-neuronx 2.14.0 (Ubuntu 24.04) SDK *",
        "venv_check": "import torch_neuronx",
    },
}

TARGET_REGIONS = {"gpu": SPOT_REGION, "trainium": SPOT_REGION,
                  "gpu-sagemaker": DEFAULT_REGION, "trainium-hyperpod": DEFAULT_REGION}
RESULTS_DIR = Path("covex-results")


def run(scripts, on, args=(), checkpoint=None, region=None, requirements=None, max_minutes=45):
    """Run one or more scripts (same folder) back to back on one machine; return the local results dir.

    on: "gpu" / "trainium" (EC2 Spot, cheapest) or "gpu-sagemaker" / "trainium-hyperpod" (on-demand).
    Each script's outputs ($SM_MODEL_DIR) land in covex-results/<job>/<script name>/.
    `checkpoint` (local folder or s3:// prefix) is mounted and passed to every script as --model <path>.
    """
    scripts = [scripts] if isinstance(scripts, (str, Path)) else list(scripts)
    folders = {Path(s).resolve().parent for s in scripts}
    if len(folders) != 1:
        raise ValueError("all scripts must be in the same folder (it is uploaded as the job's code)")
    if on not in TARGET_REGIONS:
        raise ValueError(f"on must be one of {sorted(TARGET_REGIONS)}")

    aws = _Aws(region or TARGET_REGIONS[on])
    job = f"covex-{'gpu' if on.startswith('gpu') else 'trn'}-{time.strftime('%Y%m%d-%H%M%S')}"
    prefix = f"jobs/{job}"
    aws.s3.put_object(Bucket=aws.bucket, Key=f"{prefix}/code/code.tar.gz", Body=_bundle(folders.pop()))

    args = list(args)
    model_s3 = None
    if checkpoint:
        model_s3 = checkpoint if str(checkpoint).startswith("s3://") else aws.upload_dir(checkpoint)
    reqs = requirements or DEFAULT_REQUIREMENTS
    names = [Path(s).name for s in scripts]

    if on in SPOT_TARGETS:
        out = _run_ec2_spot(aws, on, job, prefix, names, args, model_s3, reqs, max_minutes)
    elif on == "gpu-sagemaker":
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
        self.ec2 = sess.client("ec2")
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

    def ensure_instance_profile(self):
        """EC2 instance profile that can read/write this bucket. The instance terminates itself via shutdown."""
        try:
            self.iam.get_instance_profile(InstanceProfileName=EC2_ROLE_NAME)
            created = False
        except self.iam.exceptions.NoSuchEntityException:
            trust = {"Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
            self.iam.create_role(RoleName=EC2_ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust))
            self.iam.create_instance_profile(InstanceProfileName=EC2_ROLE_NAME)
            self.iam.add_role_to_instance_profile(InstanceProfileName=EC2_ROLE_NAME, RoleName=EC2_ROLE_NAME)
            created = True
        # Per-region buckets: keep the policy covering every covex bucket in the account.
        self.iam.put_role_policy(RoleName=EC2_ROLE_NAME, PolicyName="covex-s3", PolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::sagemaker-covex-neuron-{self.account}-*",
                             f"arn:aws:s3:::sagemaker-covex-neuron-{self.account}-*/*"]}]}))
        if created:
            print(f"created EC2 instance profile {EC2_ROLE_NAME}; waiting for it to propagate")
            time.sleep(15)
        return EC2_ROLE_NAME

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


# ---------------------------------------------------------------- Trainium: EC2 Spot + Neuron DLAMI

def _run_ec2_spot(aws, target, job, prefix, names, args, model_s3, reqs, max_minutes):
    spec = SPOT_TARGETS[target]
    s3_root = f"s3://{aws.bucket}/{prefix}"
    command = _job_command("/tmp/codeinput", "/tmp/out", "/tmp/model" if model_s3 else None, names, args, reqs)
    user_data = _ec2_user_data(s3_root, model_s3, command, max_minutes, spec["venv_check"])
    profile = aws.ensure_instance_profile()
    ami = _latest_ami(aws.ec2, spec["ami"])

    instance_id = launched_az = instance_type = None
    choices = [(t, *c) for t in spec["instance_types"] for c in _spot_choices(aws.ec2, t)]
    for itype, az, subnet, price in choices:
        for attempt in range(4):  # a new instance profile can take a few seconds to become usable
            try:
                resp = aws.ec2.run_instances(
                    ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
                    SubnetId=subnet, IamInstanceProfile={"Name": profile}, UserData=user_data,
                    InstanceInitiatedShutdownBehavior="terminate",
                    InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {
                        "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}},
                    BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {
                        "VolumeSize": 150, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                    TagSpecifications=[{"ResourceType": "instance", "Tags": [TAG, {"Key": "Name", "Value": job}]}],
                )
                instance_id, launched_az, instance_type = resp["Instances"][0]["InstanceId"], az, itype
                break
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if "InstanceProfile" in str(e) and attempt < 3:
                    time.sleep(10)
                    continue
                print(f"{itype} {az}: {code} - {e.response['Error']['Message'][:160]}")
                break
        if instance_id:
            print(f"launched spot {itype} {instance_id} in {az} (spot price now ${price:.3f}/hr)")
            break
    if not instance_id:
        fallback = "gpu-sagemaker" if target == "gpu" else "trainium-hyperpod"
        raise RuntimeError(f"no {target} spot capacity (or quota) in {aws.region}; try on={fallback!r}")

    started, log_key, printed, done = time.time(), f"{prefix}/log.txt", 0, False
    try:
        while time.time() < started + (max_minutes + 15) * 60:
            try:
                text = aws.s3.get_object(Bucket=aws.bucket, Key=log_key)["Body"].read().decode(errors="replace")
                print(text[printed:], end="")
                printed = len(text)
            except ClientError:
                pass
            if _exists(aws, f"{prefix}/DONE"):
                done = True
                break
            state = aws.ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
            if state["State"]["Name"] in ("shutting-down", "terminated"):
                reason = state.get("StateReason", {}).get("Message", "")
                if not _exists(aws, f"{prefix}/DONE"):
                    print(f"instance ended before finishing: {reason}")
                break
            time.sleep(15)
    finally:
        aws.ec2.terminate_instances(InstanceIds=[instance_id])  # no-op if it already terminated itself
    minutes = (time.time() - started) / 60

    out = aws.download_results(f"{prefix}/output/out.tar.gz", job)
    price = _spot_price_now(aws.ec2, instance_type, launched_az)
    _write_job_meta(out, job=job, target=target, instance=f"{instance_type} (EC2 Spot, {launched_az})",
                    status="Completed" if done else "Incomplete", approx_billed_minutes=round(minutes, 1),
                    price_per_hour=price, cost_usd=price * minutes / 60 if price else None)
    return out


def _latest_ami(ec2, name_pattern):
    images = ec2.describe_images(Owners=["amazon"], Filters=[{"Name": "name", "Values": [name_pattern]}])["Images"]
    if not images:
        raise RuntimeError(f"no AMI matching {name_pattern!r} in this region")
    return max(images, key=lambda i: i["CreationDate"])["ImageId"]


def _spot_choices(ec2, instance_type):
    """(az, default subnet, current spot $/hr) for every AZ offering the type, cheapest first."""
    offered = {o["Location"] for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}])["InstanceTypeOfferings"]}
    subnets = {s["AvailabilityZone"]: s["SubnetId"] for s in ec2.describe_subnets(
        Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]}
    choices = [(az, subnets[az], _spot_price_now(ec2, instance_type, az) or float("inf"))
               for az in offered if az in subnets]
    return sorted(choices, key=lambda c: c[2])


def _spot_price_now(ec2, instance_type, az):
    if not az:
        return None
    hist = ec2.describe_spot_price_history(InstanceTypes=[instance_type], AvailabilityZone=az,
                                           ProductDescriptions=["Linux/UNIX"], MaxResults=1)["SpotPriceHistory"]
    return float(hist[0]["SpotPrice"]) if hist else None


def _ec2_user_data(s3_root, model_s3, command, max_minutes, venv_check):
    model_sync = f"aws s3 sync --quiet {model_s3} /tmp/model" if model_s3 else ""
    return f"""#!/bin/bash
shutdown -h +{max_minutes + 10}   # hard stop: instance terminates even if everything below hangs
LOG=/var/log/covex_job.log
exec > $LOG 2>&1
( while true; do aws s3 cp --quiet $LOG {s3_root}/log.txt; sleep 15; done ) &
UPLOADER=$!
finish() {{
  tar czf /tmp/out.tar.gz -C /tmp/out . 2>/dev/null && aws s3 cp --quiet /tmp/out.tar.gz {s3_root}/output/out.tar.gz
  kill $UPLOADER; aws s3 cp --quiet $LOG {s3_root}/log.txt
  echo done | aws s3 cp --quiet - {s3_root}/DONE
  shutdown -h now
}}
trap finish EXIT
mkdir -p /tmp/out /tmp/codeinput /tmp/model
echo "[covex] $(hostname) $(ls /dev/neuron* 2>/dev/null | tr '\\n' ' ')$(nvidia-smi -L 2>/dev/null)"
# The Deep Learning AMIs ship PyTorch (CUDA or TorchNeuron) in a venv under /opt; use the one that works.
for v in /opt/aws_neuron*venv* /opt/pytorch /opt/conda/envs/pytorch; do
  if [ -x "$v/bin/python" ] && "$v/bin/python" -c {_shell_quote(venv_check)} 2>/dev/null; then
    source "$v/bin/activate"; break
  fi
done
echo "[covex] python $(command -v python)"
aws s3 cp --quiet {s3_root}/code/code.tar.gz /tmp/codeinput/code.tar.gz
{model_sync}
cd /tmp
timeout {max_minutes}m bash -c {_shell_quote(command)}
echo "[covex] exit code $?"
"""


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
