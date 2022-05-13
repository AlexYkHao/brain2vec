import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from skimage.filters import threshold_triangle
import psutil
import brainsss
import nibabel as nib

def main(args):

    logfile = args['logfile']
    directory = args['directory']
    width = 120
    printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')
    file = directory.split('/')[-1]

    #################
    ### Load Data ###
    #################

    if os.path.exists(directory):
            brain = np.asarray(nib.load(directory).get_data(), dtype='uint16')
            data_mean = np.mean(brain,axis=(0,1,2))
    else:
            printlog(F"Not found (skipping){file:.>{width-20}}")

    ##############################
    ### Output Bleaching Curve ###
    ##############################

    plt.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(10,10))
    
    xs = np.arange(len(data_mean))
    plt.plot(data_mean,color='green')
    linear_fit = np.polyfit(xs, data_mean, 1)
    plt.plot(np.poly1d(linear_fit)(xs),color='k',linewidth=3,linestyle='--')
    signal_loss = linear_fit[0]*len(data_mean)/linear_fit[1]*-100
    plt.xlabel('Frame Num')
    plt.ylabel('Avg signal')
    loss_string =  file + ' lost' + F'{int(signal_loss)}' +'%\n'
    plt.title(loss_string, ha='center', va='bottom')

    save_file = directory.split('.')[0] + '_bleaching.png'
    plt.savefig(save_file,dpi=300,bbox_inches='tight')

if __name__ == '__main__':
    main(json.loads(sys.argv[1]))