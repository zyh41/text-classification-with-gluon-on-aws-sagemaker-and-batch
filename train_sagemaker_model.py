import boto3
import json
import sagemaker as sage
import argparse

parser = argparse.ArgumentParser(description='Train language model with Amazon SageMaker.')
parser.add_argument('--image-name', type=str, default='languagemodel-ryan',
                    help='the image repository name -- set to the default in the blog')
parser.add_argument('--role-name', type=str, default='AmazonSageMaker-ExecutionRole-{xxx}', # edit here
                    help='the SakeMaker execution role name -- set to the default in the blog')
parser.add_argument('--region-name', type=str, default='us-west-2',
                    help='us-east-1, us-east-2, us-west-2, eu-west-1 -- set to the default in the blog')
args = parser.parse_args()

SAGEMAKER_REGIONS = {'us-east-1', 'us-east-2', 'us-west-2', 'eu-west-1'}

if __name__ == '__main__':

    # start a sagemaker session
    sess = sage.Session()

    # Get our account-id and our region
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name

    # override region to blog default if not in acceptable location
    if region not in SAGEMAKER_REGIONS:
        region = args.region_name

    # Get our image URI and the role we created in our CloudFormation Template
    image = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account, region, args.image_name)
    role ='arn:aws:iam::{}:role/{}'.format(account, args.role_name)

    # Create a sagemaker training instance using our image URI
    languagemodel = sage.estimator.Estimator(image, role, 1, "ml.p2.xlarge",
                           output_path="s3://{}/output".format(sess.default_bucket()),
                           sagemaker_session=sess)

    # read our local config information
    config = json.load(open('config/config.json'))

    # set our training configuration for the model
    languagemodel.hyperparam_dict = config

    # upload our training data to s3 
    # the output will be something like this:
    # s3://sagemaker-us-east-1-{account-id}/data/train.csv'
    data_location = sess.upload_data(path="data/train.csv")

    # finally we fit our data - sit back and read the stream!
    languagemodel.fit(data_location)

