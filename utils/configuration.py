import os
import shutil
import time
import argparse
import sys
import itertools


int_args = [ "nr_layers_gen", "nr_heads_gen", "model_dim_gen",
             "nr_layers_dis", "nr_heads_dis", "model_dim_dis",
             "epochs", "batch_size" ]

float_args = [ "lr_gen", "lr_dis"]

bool_args = [ "shuffle_data" ]


def fetch_conf_parameters_noargs():



    # read configuration file
    conf_file = open("used_configuration.txt", 'r')
    conf_content = conf_file.readlines()
    configuration = {}

    for line in conf_content:
        # remove end lines
        line = line.strip()

        if line.startswith('#'):
            continue

        tmp = line.split("=")
        key, value = tmp[0], tmp[1]

        # print(key,value)

        # parse arguments
        values = value.split(',')
        if key in int_args:
            values = [int(item) for item in values]
        elif key in float_args:
            values = [float(item) for item in values]
        elif key in bool_args:
            values = [bool(item) for item in values]
        else:
            values = [str(item) for item in values]
        configuration[key] = values

    # print(configuration)
    parameters = list(configuration.keys())
    # print(parameters)
    #  create all possible configurations for
    all_configurations = list(itertools.product(*list(configuration.values())))
    # nr_confs = len(list(all_configurations))
    # print(f"Number of configurations:{nr_confs}")

    return parameters, all_configurations


def fetch_conf_parameters():

    # read configuration parameters
    conf_file = "net_conf.txt"

    with open( conf_file ) as f:
        conf_content = f.readlines()
    configuration = {}

    for line in conf_content:
        # remove end lines
        line = line.strip()

        if line.startswith('#'):
            continue

        tmp = line.split("=")
        key, value = tmp[0], tmp[1]

        # print(key,value)

        # parse arguments
        values = value.split(',')
        if key in int_args:
            values = [int(item) for item in values]
        elif key in float_args:
            values = [float(item) for item in values]
        elif key in bool_args:
            values = [bool(item) for item in values]
        else:
            values = [str(item) for item in values]
        configuration[key] = values

    # print(configuration)
    parameters = list(configuration.keys())
    # print(parameters)
    #  create all possible configurations for
    all_configurations = list(itertools.product(*list(configuration.values())))
    # nr_confs = len(list(all_configurations))
    # print(f"Number of configurations:{nr_confs}")

    return parameters, all_configurations



if __name__ == "__main__":

    parameters, configurations = fetch_conf_parameters()
    for conf in configurations:
        tmp = list(zip(parameters,conf))
        tmp_conf = { item:val for item,val in tmp }
        print(tmp_conf)
