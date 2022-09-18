import numpy as np
import pandas as pd

class BVHWriter():
    def __init__(self):
        pass
    
    def write(self, X, ofile):
        
        # Writing the skeleton info
        ofile.write('HIERARCHY\n')
        
        self.motions_ = []
        self._printJoint(X, X.root_name, 0, ofile)

        # Writing the motion header
        ofile.write('MOTION\n')
        ofile.write('Frames: %d\n'%X.values.shape[0])
        ofile.write('Frame Time: %f\n'%X.framerate)

        # Writing the data
        self.motions_ = np.asarray(self.motions_).T
        better_lines = []
        for tmp in self.motions_:
            line = [ "{:.6f}".format(el) for el in tmp ]
            line[ -1 ] = line[-1] + ' '
            better_lines.append(line )

        better_lines = [ " ".join( item )  for item in better_lines ]

        # lines = [" ".join(item) for item in self.motions_.astype(str)]
        # ofile.write("".join("%s\n"%l for l in lines))
        ofile.write("".join("%s\n" % l for l in better_lines))

    def _printJoint(self, X, joint, tab, ofile):
        
        if X.skeleton[joint]['parent'] == None:
            ofile.write('ROOT %s\n'%joint)
        elif len(X.skeleton[joint]['children']) > 0:
            ofile.write('%sJOINT %s\n'%('\t'*(tab), joint))
        else:
            ofile.write('%sEnd Site\n'%('\t'*(tab)))

        ofile.write('%s{\n'%('\t'*(tab)))
        
        ofile.write('%sOFFSET %3.6f %3.6f %3.6f\n'%('\t'*(tab+1),
                                                X.skeleton[joint]['offsets'][0],
                                                X.skeleton[joint]['offsets'][1],
                                                X.skeleton[joint]['offsets'][2]))
        channels = X.skeleton[joint]['channels']
        n_channels = len(channels)

        if n_channels > 0:
            for ch in channels:
                self.motions_.append(np.asarray(X.values['%s_%s'%(joint, ch)].values))

        if len(X.skeleton[joint]['children']) > 0:
            ch_str = ''.join(' %s'*n_channels%tuple(channels))
            ofile.write('%sCHANNELS %d%s\n' %('\t'*(tab+1), n_channels, ch_str))

            for c in X.skeleton[joint]['children']:
                self._printJoint(X, c, tab+1, ofile)

        ofile.write('%s}\n'%('\t'*(tab)))
