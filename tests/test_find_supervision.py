import os

def find_supervision( dataset_root_dir = "../dataset/release_bvh/"):


    ### creates a dic dataset that stores: motions, character_names, motion_type, skeletan structure(t-pose) and joint names


    characters = {}
    motion_types = {}


    ### iterate the filesystem
    for root, dirs, files in os.walk(dataset_root_dir, topdown=False):

        print(f"Parsing folder:{root}")
        character = root.split('/')[-1]
        print(f"Character:{character}")

        tmp = []

        for name in files:
            if name.endswith(".bvh"):

                # create name of bvh file to process
                bvh_file = os.path.join(root, name)
                print( f"Parsing bvh file:{bvh_file}" )
                print(f"Parsing bvh file:{name}")
                tmp.append(name)

        characters[ character ] = tmp



    for character in characters:

        motion_types = characters[character]
        common_motions = []
        for character_new in characters:

            if character == character_new:
                continue

            motion_types_new  = characters[character_new]

            common_motions = list( set(motion_types).intersection(motion_types_new))
            if len(common_motions) > 0:
                print( f"Character1:{character}, Character2:{character_new}" )
                print( common_motions )
                print(len(common_motions))
            r = 1




if __name__ == "__main__":

    find_supervision()