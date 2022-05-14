import time
import sys
import os
import re
import json
import datetime
import pyfiglet
import textwrap
import brainsss

modules = 'gcc/6.3.0 python/3.6.1 py-numpy/1.14.3_py36 py-pandas/0.23.0_py36 viz py-scikit-learn/0.19.1_py36 antspy/0.2.2'
channels = ['red', 'green', 'anat', 'anat_green']

class FlyBatcher(object):

    def __init__(self, config, channels=channels):
        super().__init__()
        self.channels = channels
        with open(config) as json_file:
            self.config = json.load(json_file)
        self.setup()
    

    def setup(self):
        required_keys = ['print_width', 'fly_list', 'nodes', 'nice', 'scripts_path',
         'com_path', 'dataset_path']
        assert set(required_keys) <= set(self.config.keys())
        for key in required_keys:
            setattr(self, key, self.config[key])

        #########
        # unpack all flies
        #########
        self.flies = self.fly_list
        self.width = self.print_width 
        #########
        # set up log file
        #########
        self.logfile = os.path.join(self.dataset_path, 'log_'+time.strftime("%Y%m%d-%H%M%S") + '.txt')
        self.printlog = getattr(brainsss.Printlog(logfile=self.logfile), 'print_to_log')
        sys.stderr = brainsss.Logger_stderr_sherlock(self.logfile)
        self.printlog(sys.version)

        #########
        # print title
        #########
        title = pyfiglet.figlet_format("Brainsss", font="cyberlarge" ) #28 #shimrod
        title_shifted = ('\n').join([' '*28+line for line in title.split('\n')][:-2])
        self.printlog(title_shifted)
        day_now = datetime.datetime.now().strftime("%B %d, %Y")
        time_now = datetime.datetime.now().strftime("%I:%M:%S %p")
        self.printlog(F"{day_now+' | '+time_now:^{self.width}}")
        self.printlog("")        

    
    def pre_processing(self):
        #self.bleach_curve()
        #self.mean_brain()
        self.motion_correction()


    def submit_jobs(self, job_ids):
        logfile = self.logfile
        com_path = self.com_path
        for job_id in job_ids:
            brainsss.wait_for_job(job_id, logfile, com_path)

    
    def bleach_curve(self):
        self.printlog(f"\n{'   BLEACHING QC   ':=^{self.width}}")
        job_ids = []
        for fly in self.flies:
            for channel in self.channels:
                if not fly[channel]:
                    continue
                directory = os.path.join(self.dataset_path, fly['id'], fly[channel])
                args = {'logfile': self.logfile, 'directory': directory}
                script = 'bleaching.py'
                job_id = brainsss.sbatch(jobname='bleachqc',
                         script=os.path.join(self.scripts_path, script),
                         modules=modules,
                         args=args,
                         logfile=self.logfile, time=1, mem=1, nice=self.nice, nodes=self.nodes)
                job_ids.append(job_id)

        self.submit_jobs(job_ids)


    def mean_brain(self):
        self.printlog(f"\n{'   MEAN BRAINS   ':=^{self.width}}")
        job_ids = []
        for fly in self.flies:
            for channel in self.channels:
                if not fly[channel]:
                    continue
                directory = os.path.join(self.dataset_path, fly['id'], fly[channel])
                args = {'logfile': self.logfile, 'directory': directory}
                script = 'make_mean_brain.py'
                job_id = brainsss.sbatch(jobname='meanbrn',
                         script=os.path.join(self.scripts_path, script),
                         modules=modules,
                         args=args,
                         logfile=self.logfile, time=1, mem=1, nice=self.nice, nodes=self.nodes)
                job_ids.append(job_id)
        
        self.submit_jobs(job_ids)

    def motion_correction(self):
        self.printlog(f"\n{'   MOTION CORRECTION   ':=^{self.width}}")
        job_ids = []

        for fly in self.flies:
            carrier_channel = 'red'
            passenger_channel = 'green'
            step_size = 100  # number of volumes
            mem = 4
            time_moco = 12

            directory = os.path.join(self.dataset_path, fly['id'])
            carrier = os.path.join(self.dataset_path, fly['id'], fly[carrier_channel])
            passenger = os.path.join(self.dataset_path, fly['id'], fly[passenger_channel])

            args = {'logfile': self.logfile, 'directory': directory, 'carrier': carrier, 'passenger': passenger,
                    'step_size': step_size}
            script = 'moco.py'
            job_id = brainsss.sbatch(jobname='brain2vec',
                                     script=os.path.join(self.scripts_path, script),
                                     modules=modules,
                                     args=args,
                                     logfile=self.logfile, time=time_moco, mem=mem, nice=self.nice, nodes=self.nodes)

            job_ids.append(job_id)

        self.submit_jobs(job_ids)



'''

##################
### Start MOCO ###
##################
timepoints = 10 #number of volumes
step = 5 #how many volumes one job will handle
mem = 4
time_moco = 3

printlog(f"\n{'   MOTION CORRECTION   ':=^{width}}")
# This will immediately launch all partial mocos and their corresponding dependent moco stitchers
stitcher_job_ids = []
progress_tracker = {}
for fly in flies:
    directory = os.path.join(dataset_path, fly)
    fly_print = directory.split('/')[-1]

    moco_dir = os.path.join(directory, 'moco')
    if not os.path.exists(moco_dir):
        os.makedirs(moco_dir)

    starts = list(range(0,timepoints,step))
    stops = starts[1:] + [timepoints]

    #######################
    ### Launch partials ###
    #######################
    job_ids = []
    for start, stop in zip (starts, stops):
        args = {'logfile': logfile, 'directory': directory, 'start': start, 'stop': stop}
        script = 'moco_partial.py'
        job_id = brainsss.sbatch(jobname='moco',
                             script=os.path.join(scripts_path, script),
                             modules=modules,
                             args=args,
                             logfile=logfile, time=time_moco, mem=mem, nice=nice, silence_print=True, nodes=nodes)
        job_ids.append(job_id)

    printlog(F"| moco_partials | SUBMITTED | {fly_print} | {len(job_ids)} jobs, {step} vols each |")
    job_ids_colons = ':'.join(job_ids)
    for_tracker = '/'.join(directory.split('/')[-2:])
    progress_tracker[for_tracker] = {'job_ids': job_ids, 'total_vol': timepoints}

    #################################
    ### Create dependent stitcher ###
    #################################

    args = {'logfile': logfile, 'directory': moco_dir}
    script = 'moco_stitcher.py'
    job_id = brainsss.sbatch(jobname='stitch',
                         script=os.path.join(scripts_path, script),
                         modules=modules,
                         args=args,
                         logfile=logfile, time=2, mem=4, dep=job_ids_colons, nice=nice, nodes=nodes)
    stitcher_job_ids.append(job_id)

if bool(progress_tracker): #if not empty
    brainsss.moco_progress(progress_tracker, logfile, com_path)

for job_id in stitcher_job_ids:
    brainsss.wait_for_job(job_id, logfile, com_path)

###############
### Z-Score ###
###############

printlog(f"\n{'   Z-SCORE   ':=^{width}}")
job_ids = []
for fly in flies:
    directory = os.path.join(dataset_path, fly)
    args = {'logfile': logfile, 'directory': directory, 'smooth': False, 'colors': ['green']}
    script = 'zscore.py'
    job_id = brainsss.sbatch(jobname='zscore',
                         script=os.path.join(scripts_path, script),
                         modules=modules,
                         args=args,
                         logfile=logfile, time=2, mem=4, nice=nice, nodes=nodes)
    job_ids.append(job_id)

for job_id in job_ids:
    brainsss.wait_for_job(job_id, logfile, com_path)

############
### Done ###
############

time.sleep(30) # to allow any final printing
day_now = datetime.datetime.now().strftime("%B %d, %Y")
time_now = datetime.datetime.now().strftime("%I:%M:%S %p")
printlog("="*width)
printlog(F"{day_now+' | '+time_now:^{width}}")
'''
