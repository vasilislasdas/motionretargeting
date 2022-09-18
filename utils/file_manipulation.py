import os
import sys
sys.path.append( '../others/PyMO')
from pymo.parsers import BVHParser

parser = BVHParser()

#

bvh_files = []
dataset_root_dir = "../dataset/release_bvh/"
# dataset_root_dir = "../dataset/release_bvh/BigVegas"

for root, dirs, files in os.walk(dataset_root_dir, topdown = False):

   print( f"Parsing folder:{root}" )
   r = 1
   # if "BigVegas" not in root:
   #     print("FOUND BIG VEGAS FILE!")
   #     continue
   for name in files:
      if name.endswith(".bvh"):
          tmp = os.path.join(root, name)
          print( f"Parsing file:{tmp}" )

          # store the name of the bvh file
          bvh_files.append( tmp )

          # try to read the files one after the other for test
          # parsed_data = parser.parse( tmp )

with open('your_file.txt', 'w') as f:
    for item in bvh_files:
        f.write("%s\n" % item)



print( f"{len(bvh_files)} were parsed succesfuly!")




