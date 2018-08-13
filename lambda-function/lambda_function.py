#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import boto3
import traceback
import json
import os
import time

# Environment variables to make batch work - or make this work automatically
ENVIRONMENTNAME = os.environ['ENVIRONMENTNAME']
JOBQUEUENAME = os.environ['JOBQUEUENAME']
JOBNAME = os.environ['JOBNAME']

########################################################################################
########################################################################################

def lambda_handler(event, context):
    """
    This function is a lambda_handler that kicks off a containerized
    job using aws-batch

    Args:
    -----
    event: dict
        the data passed to the lambda_handler that we'll use
        to parse the bucket string
    context: LambdaContext object
        the lambdacontext object as part of aws-lambda-functions
    Returns:
    --------
    response_sj: dict
        the creation response that describes the job submitted to 
        the most recent job queue
    """
    try:
        client = boto3.client('batch', region_name='us-east-1') #RYAN

        ce = client.describe_compute_environments(
                                computeEnvironments=[ENVIRONMENTNAME]
                                )
        compute_environment = ce['computeEnvironments'][0]

        jq = client.describe_job_queues(
                                    jobQueues = [JOBQUEUENAME]
                                    )
        job_queue = jq['jobQueues'][0]

        job_definitions = client.describe_job_definitions(jobDefinitionName = JOBNAME,
                                                    status = 'ACTIVE')['jobDefinitions']
        #garantees most recent revision of your job definition
        job_definition = sorted(job_definitions, key = lambda x: x['jobDefinitionArn'])[::-1][0]

        response_sj = client.submit_job(jobName = JOBNAME, 
                                        jobQueue = JOBQUEUENAME,
                                        jobDefinition = job_definition['jobDefinitionArn'])

        # add information to the submit job status for the teardown function
        response_sj['computeEnvironmentArn'] = compute_environment['computeEnvironmentArn']
        response_sj['jobDefinitionArn'] = job_definition['jobDefinitionArn']
        response_sj['jobQueueArn'] = job_queue['jobQueueArn']

        return response_sj

    except Exception as e:
        traceback.print_exc()
        raise e
