import subprocess
import re
try:
    import progressbar
except ModuleNotFoundError:
    print("installing dependencies")
    install = subprocess.Popen("source ~/.bashrc && pyenv activate py3 && which pip && pip install progressbar2", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    install.wait()
    import progressbar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from SDDS import readSDDS
m_e = physical_constants["electron mass energy equivalent in MeV"][0]
from ipywidgets import interact, interactive, widgets
from time import sleep

def error_scan(max_iter, phase_error, amplitude_error, lattice='BC1LINE', process=4):
    """
    Run and track progress of error scan in elegant
    """
    assert lattice == 'BC1LINE' or lattice == 'BC2LINE', "lattice must be \'BC2LINE\' or \'BC1LINE\'"
    macro_input = 'iter={{}},phase_error={},amplitude_error={},lattice_name={}'.format(phase_error,amplitude_error,lattice)

    bar = progressbar.ProgressBar(max_value=max_iter, redirect_stderr=False)
    all_processes = process_setup(macro_input, max_iter, process)

    index = [0] * len(all_processes)
    perc = 0.0
    while all_processes:
        # Check all process status
        for i, p in enumerate(all_processes):
            line = p.stdout.readline().decode('utf-8')
#             with open('log_{}.txt'.format(i), 'a+') as f:
#                 f.write(line)
            l1 = line.find("tracking step")
            if l1 > -1:
                index[i] = int(re.findall(r'\d+', line)[0])
            if line == '' and p.poll() != None:
                # Process is finished
                try:
                    # If this works there was an error
                    line = p.stderr.readlines().decode('utf-8')
                    print("ERROR OCCURRED:")
                    for ln in line:
                        print(ln)
                    all_processes.pop(i)
                except:
                    # Process finished successfully
                    all_processes.pop(i)
                    # print("Completed without error")
        # Print progress
        if np.sum(index) >  perc:
            perc = np.sum(index).astype(int)
            bar.update(perc)
        sleep(0.003)  # can't go too fast for progressbar to keep up
    bar.update(max_iter)
    concat = subprocess.Popen("sddscombine error_*.log error_all.log", shell=True)
    concat.wait()
    conv = subprocess.Popen("sddsconvert -ascii error_all.log", shell=True)
    conv.wait()
    concat2 = subprocess.Popen("sddscombine run_setup_*.output.sdds run_setup_all.output.sdds", shell=True)
    concat2.wait()
    # Remove embedded string parameters
    edit = subprocess.Popen("sddsprocess run_setup_all.output.sdds -delete=par,Filename,SVNVersion", shell=True)
    edit.wait()


def process_setup(macro, iter_count, process_count):
    processes = []

    for i in range(process_count)[::-1]:
        local_iter = (iter_count + i) // process_count
        local_macro = macro.format(local_iter)
        p = subprocess.Popen("elegant error_elegant.ele -macros=index={},{}".format(i, local_macro),
                             shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p)

    return processes

def read_settings(fn):
    """
    Read a error log file from an elegant error study and produce a DataFrame with Elements and settings for each step.
    fn (str): Log file name.
    """
    with open(fn) as openf:
        all_lines = openf.readlines()
        total_length = len(all_lines)
        for i, line in enumerate(all_lines):
            if line.find('0 ') == 0:
                start = i
            if line.find('1 ') == 0:
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



def read_settings3(fn):
    """
    Read a error log file from an elegant error study and produce a DataFrame with Elements and settings for each step.
    fn (str): Log file name.
    """
    # Account for page number and row count
    parameters = 2
    with open(fn) as openf:
        all_lines = openf.readlines()
        total_length = len(all_lines)
        for i, line in enumerate(all_lines):
            if line == '! page number 1\n': #14
                start = i
            elif line == '! page number 2\n': #26
                step = i
            elif line.find("&parameter") == 0:
                parameters += 1
    # Number of lines in each page
    skip_period = step - start
    # Number of data points in each page // non-data lines hard-coded to 6
    data_period = skip_period - parameters
    # -1 to more back to end of header
    total_steps = (total_length - start) //  skip_period
    skips = np.arange(start, start + parameters).reshape(-1, 1) + skip_period * np.arange(total_steps).reshape(1, -1)
    skips = skips.flatten()
    skips = np.concatenate([np.arange(start).flatten()] + [skips])
    
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
            fig, ax = plt.subplots(2, 2, constrained_layout=False, figsize=(10, 10),
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
            
            ax[1, 0].set_xlabel("Position (ps)")
            ax[0, 1].set_ylabel("Energy (MeV)")
            
#             ax[1, 1].table(cellText=np.arange(4).reshape(2,2), loc='center', rowLabels=settings.columns.levels[1])
            ax[1, 1].set_axis_off()
            self.format_table(step, ax[1,1])
            fig.tight_layout() 
