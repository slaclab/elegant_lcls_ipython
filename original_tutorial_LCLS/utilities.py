import subprocess
import re
#import progressbar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
#from rsbeams.rsdata.SDDS import readSDDS
from SDDS import readSDDS
m_e = physical_constants["electron mass energy equivalent in MeV"][0]
from ipywidgets import interact, interactive, widgets

def error_scan(max_iter, phase_error, amplitude_error, lattice='BC1LINE'):
    """
    Run and track progress of error scan in elegant
    """
    assert lattice == 'BC1LINE' or lattice == 'BC2LINE', "lattice must be \'BC2LINE\' or \'BC1LINE\'"
    macro_input = 'iter={},phase_error={},amplitude_error={},lattice_name={}'.format(max_iter,phase_error,amplitude_error,lattice)

    #bar = progressbar.ProgressBar(max_value=max_iter)
    p = subprocess.Popen("elegant error_elegant.ele -macros={}".format(macro_input),
     shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('starting')
    while True:
        line = p.stdout.readline()
        # print(line)
        l1 = line.find("tracking step")
        if l1 > -1:
            index = int(re.findall(r'\d+', line)[0])
            # print(index)
            #if index % int(max_iter * 0.05) == 0:
                #perc = int(index / float(max_iter) * max_iter)
                #bar.update(perc)
        if line == '' and p.poll() != None:
            try:
                line = p.stderr.readlines()
                for ln in line:
                    print(ln)
            except:
                print("Completed without error")
            break

def read_settings(fn):
    """
    Read a error log file from an elegant error study and produce a DataFrame with Elements and settings for each step.
    fn (str): Log file name.
    """
    with open(fn) as openf:
        all_lines = openf.readlines()
        total_length = len(all_lines)
        for i, line in enumerate(all_lines):
            if line.find('0') == 0:
                start = i
            if line.find('1') == 0:
                step = i
                break
    skip_period = step - start
    # -3 accounts for 2 step lines and 1 blank
    data_period = step - start - 3
    total_steps = (total_length - start) //  skip_period
    
    skips = [np.arange(start, total_length + 1, skip_period), np.arange(start + 1, total_length + 1, skip_period), np.arange(start + data_period + 2, total_length + 1, skip_period)]   
    skips = np.concatenate([np.arange(start + 2)] + skips)
    test = pd.read_csv(fn, delim_whitespace=True, error_bad_lines=False, skiprows=skips.flatten(), header=None)
    if (data_period) == (test.sort_values(by=[3]).values[data_period-1::total_steps, [2, 3]].shape[0]):
        frame = pd.DataFrame(test.values[:, 0].reshape(total_steps, data_period), 
                         columns=pd.MultiIndex.from_arrays(np.flip(np.flip(test.sort_values(by=[3]).values[data_period-1::total_steps, [2, 3]].T, 0),1), names=['Element', 'Attribute']),
                         index=np.arange(total_steps))

    else:
        frame = pd.DataFrame(test.values[:, 0].reshape(total_steps, data_period))
        print('Please add more "data_points" to cover all parameters changes. Still missing:')
        print((data_period) - (test.sort_values(by=[3]).values[data_period-1::total_steps, [2, 3]].shape[0]))    
    return frame


class lps_plot():
    
    def __init__(self, dataset, settings):
        self.dataset = dataset
        self.settings = settings
        
    def __call__(self):
        
#         temp_formatter = np.arange(800, 6001, 100)
#         temp_control = widgets.IntSlider(value=2000, min=800, max=6001, step=100)
        page_control = widgets.IntText()
        _ = interact(self.plot, step=page_control)
        
    def format_table(self, step, ax):
        row_names = self.settings.columns.levels[0]
        col_names = self.settings.columns.levels[1]
        data = self.settings.iloc[step, :].values.reshape(col_names.shape[0], row_names.shape[0]).T
        table = ax.table(cellText=data, loc='right', rowLabels=row_names, colLabels=col_names, colWidths=[1,1,1])
        
        return table
        
    def plot(self, step):
        if step < 0 or step > self.dataset.shape[0]:
            print("Invalid Page number. Must be between 0 and {}".format(self.dataset.shape[0]))
        else:
            gridspec = dict(width_ratios=[5, 1], height_ratios=[5, 1])
            fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 10),
                                  gridspec_kw=gridspec)
            # 2D Histogram
            ax[0, 0].hist2d((self.dataset[step, :, 4] - np.average(self.dataset[step, :, 4])) * 1e12, 
                             self.dataset[step, :, 5] * m_e, bins=(64, 64))
            # Energy Histogram
            bins, edges = np.histogram(self.dataset[step, :, 5] * m_e, bins=64)
            centers = [(edges[i] + edges[i + 1]) / 2. for i in range(len(edges) - 1)]
            ax[0, 1].plot(bins / np.max(bins).astype(float), centers)
            ax[0, 1].invert_xaxis()
            ax[0, 1].yaxis.set_label_position("right")
            ax[0, 1].yaxis.tick_right()
            
            # Position Hisogram
            bins, edges = np.histogram((self.dataset[step, :, 4] - np.average(self.dataset[step, :, 4])) * 1e12, bins=64)
            centers = [(edges[i] + edges[i + 1]) / 2. for i in range(len(edges) - 1)]
            ax[1, 0].plot(centers, bins / np.max(bins).astype(float))
            
            ax[1, 0].set_xlabel("Position (ps)",size=14)
            ax[0, 1].set_ylabel("Energy (MeV)",size=14)
            
#             ax[1, 1].table(cellText=np.arange(4).reshape(2,2), loc='center', rowLabels=settings.columns.levels[1])
            ax[1, 1].set_axis_off()
            self.format_table(step, ax[1,1])
            fig.tight_layout() 
