#!/usr/bin/env bash
# Launch the single round-1 trn2.48xlarge Spot instance with a hard self-terminate (STOP_MIN, default 120 min).
# Usage: AWS_PROFILE=<profile> REGION=us-east-2 KEY_NAME=<ec2 key> SUBNET_ID=<subnet> SG_ID=<sg> OWNER=<name> \
#        benchmarks/scripts/launch_round1.sh [--dry-run]
# Nothing here runs without an explicit call. Spend is capped by STOP_MIN (default 120 min, about $17 at $8.60/h).
set -euo pipefail
REGION=${REGION:-us-east-2}; DRY=${1:-}
AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI Neuron (Ubuntu 24.04)*" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "AMI: $AMI"
USERDATA=$(cat <<UD
#!/bin/bash
shutdown -h +${STOP_MIN:-120}   # hard stop (default 2 h; round 1 uses STOP_MIN=360); instance-initiated shutdown = terminate
UD
)
aws ec2 run-instances --region "$REGION" $DRY \
  --image-id "$AMI" --instance-type trn2.48xlarge \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --instance-initiated-shutdown-behavior terminate \
  --key-name "$KEY_NAME" --subnet-id "$SUBNET_ID" --security-group-ids "$SG_ID" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data "$USERDATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=covex-2x-round1},{Key=Project,Value=covex-neuron},{Key=Experiment,Value=2x-round1},{Key=Owner,Value=${OWNER}},{Key=AutoTerminate,Value=${STOP_MIN:-120}min}]" \
                       "ResourceType=volume,Tags=[{Key=Project,Value=covex-neuron},{Key=Experiment,Value=2x-round1}]" \
  --query 'Instances[0].InstanceId' --output text
