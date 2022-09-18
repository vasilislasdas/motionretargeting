import os
import shutil
import time
import argparse
import sys


class NetParser( argparse.ArgumentParser ):

    def __init__(self, description):
        super().__init__(description=description)
        self.examples = str("python complex_systems.py --rule1 0 --rule2 22 \n"
                             "python complex_systems.py --rule1 128 --rule2 10 \n"
                             "python complex_systems.py --rule1 204 --rule2 0  -t 4\n\n"
                            "All results are written in a folder where the script lies with a very long name.\n\n" )


    def format_help(self):

        formatter = self._get_formatter()

        # description
        formatter.add_text( 'Decription:' + self.description)

        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        #epilog
        formatter.add_text(self.epilog)
        formatter.add_text('\n')

        # determine help from format above
        return formatter.format_help()


    def error(self, message ):

        if message:
            sys.stderr.write( "ERROR!! {} !!\n\n".format( message ) )
        self.print_help()
        sys.stderr.write("\n\nExamples:\n{}".format(self.examples))
        sys.exit(2)


def fetchParameters( args ):

    # all parameters of the program needed, reguired + optional
    rule1 = args[ "rule1" ] #  required
    rule2 = args[ "rule2" ] #  required
    nr_cells = args[ "cells" ] # has default
    nr_steps = args[ "steps" ] # has default
    nr_threads = args[ "nrthreads" ] # has default 1
    init = args[ "init" ] # None is acceptable
    lam = args[ "lambda" ]  # None is acceptable
    visual = args[ "plot" ]

    return ( rule1, rule2, nr_cells, nr_steps, nr_threads, init, lam, visual )



def printArgs( args ):
    print("----------INPUT PARAMETERS----------")
    print("Rule1:{}".format(args[0] ) )
    print("Rule2:{}".format(args[1]))
    print("Number of cells:{}".format( args[2] ) )
    print("Number of steps:{}".format( args[3] ) )
    print("Number of threads:{}".format( args[4] ) )
    print("Initial conditions:{}".format(args[5]))
    print("Diploid mixture probability:{}\n".format(args[6]))


def setupAgs():

    parser = NetParser('Motion retargeting parser')

    requiredArgumements = parser.add_argument_group('Required arguments')
    optionalArguments = parser.add_argument_group('Optional arguments')

    # parameters needed for the contruction of the nets+training
    # datasetfile #
    # nr_layers = 4
    # nr_heads = 8
    # model_dim = 128
    # device = 'cuda' # done
    # epochs = 120 # done
    # batch_size = 120 # done
    # shuffle_data = True # done


    ##### reguired arguments
    requiredArgumements.add_argument('--dataset', type=str, required=True, metavar='DATASET',
                                     help='File contating the preprocessed training data')

    requiredArgumements.add_argument( '--epochs', type=int, required=True, metavar='EPOCHS',
                                     help='Number of epochs for training')

    requiredArgumements.add_argument( '--device', type=str, required=True, metavar='DEVICE',
                                     help='Compute device: cpu or cuda')

    requiredArgumements.add_argument( '--batch', type=int, required=True, metavar='BATCH',
                                     help='Number of samples per epoch for each class. Note that the total samples is: x25')

    requiredArgumements.add_argument( '--shufle', type=bool, required=True, metavar='SHUFFLE',
                                     help='Shuffle training data for training: True, False')


    requiredArgumements.add_argument( '--rule1', type=int, required=True, metavar='RULE1_NUMBER',
                                     help='First rule(between 0-255)')
    requiredArgumements.add_argument( '--rule2', type=int, required=True, metavar='RULE2_NUMBER',
                                     help='Second rule(between 0-255)')

    # optional arguments
    optionalArguments.add_argument( '--cells', type=int, metavar='CELLS', default=1000, required=False,
                                   help='Number of cells of 1D CA. Default is 1000 cells.')
    optionalArguments.add_argument( '--steps', type=int, metavar='TIMESTEPS', default=500, required=False,
                                   help='Number of timesteps for the evolution of the CA. Default is 500 timesteps.')

    optionalArguments.add_argument( '-t', '--nrthreads', type=int, metavar='NR_THREADS', required=False, default=1,
                                   help='Number of processes. Default behaviour is single-core process. Suggested number: 2 * processors - 2.'
                                        ' E.g. for a quad-core processor with twice virtual threads: 2 * 4 - 2 = 6')

    optionalArguments.add_argument( '--init', type=float, metavar='INITIAL_CONDITIONS_PROB', required=False,
                                   help='Specifies the initial condition of the CA: prob of distribution {0,1}. '
                                        'Allowed values [0,1], Default behaviour is a single 1 in the middle.' )

    optionalArguments.add_argument( '-l', '--lambda', type=str, metavar='λ', required=False,
                                   help='Mixture probability of diploid-ECA for a specific value of lambda: (1-λ)f1 + λ * f2. '
                                        'For 0, rule 1 is retrieved. For λ = 1, rule2 is retrieved. Default is the list of '
                                        'values from the  assignment.')

    optionalArguments.add_argument('-p', '--plot', type=str, metavar='PLOT_RESULTS', required=False,
                                   help='Plot results. on/off, true/false. Default is on')


    return parser