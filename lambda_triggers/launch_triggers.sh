#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
MYUNIQUENAME=$1

if [ "$MYUNIQUENAME" == "" ]
then
    echo "Usage: $0 <MYUNIQUENAME-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

echo '{"LambdaFunctionConfigurations": [
    {
      "Id": "predict",
      "LambdaFunctionArn": "arn:aws:lambda:'$region':'$account':function:awsbatch-lambda-predict-function",
      "Events": ["s3:ObjectCreated:*"]
      }
    ]
}' >> predict.json

aws s3api put-bucket-notification-configuration --bucket $MYUNIQUENAME-predict-bucket --notification-configuration file://predict.json

echo "SUCCESS!!"





