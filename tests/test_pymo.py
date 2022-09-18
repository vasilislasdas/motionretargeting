import os
import sys
sys.path.append( '../others/PyMO')
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from pymo.writers import BVHWriter

# create bvh parser
parser = BVHParser()

# input bvh
file="tmp.bvh"

# read input from file
mocap_data = parser.parse( file )

# fetch motion
motion_in_euler = mocap_data.values

# do something with the motion:
modified_motion = motion_in_euler - 0.1

# put the values back
mocap_data.values = modified_motion

# output file to store the new bvh file
outfile = "modified.bvh"
f = open( outfile, 'w')

# create writer
writer = BVHWriter()

#  write stuff back
writer.write( mocap_data, f )




