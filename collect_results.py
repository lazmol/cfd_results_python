#!/usr/bin/env python3
'''Simple python script to collect and analyise cfd results logged into several files'''
import glob
import os
import re
import numpy as np
import itertools as it
import pandas as pd
import matplotlib as mpl
mpl.use('ps')  # use ps backend
import matplotlib.pyplot as plt


__author__ = 'Laszlo Molnar'
__version__ = '0.1.0'
__email__ = 'lazlowmiller@gmail.com'


def find_out_files(pattern='**/*.out'):
    '''Find output files'''
    return sorted(glob.glob(pattern, recursive=True))


class OutFile():
    '''Class to read/handle data from out file'''
    def __init__(self, out_file):
        self.out_file = out_file
        self.find_model_version()
        self.find_load_case()
        self.read_file()
        self.read_ambient_data()
        self.read_hx_data()
        self.read_comp_temps()

    def find_model_version(self):
        path_pieces = self.out_file.split(os.sep)
        m = re.search(r'^v\d+$', '\n'.join(path_pieces), re.MULTILINE | re.IGNORECASE)
        if m:
            self.model_ver = m.group()
        else:
            print('WARNING: model version could not be found from path:', self.out_file)
            self.model_ver = np.nan

    def find_load_case(self):
        path_pieces = self.out_file.split(os.sep)
        lc_patterns = ['TT', 'TS']
        pattern = r'^({lcs})[-_]?\d+'.format(lcs='|'.join(lc_patterns))
        m = re.search(pattern, '\n'.join(path_pieces), re.MULTILINE | re.IGNORECASE)
        if m:
            load_case = re.sub('[-_]', '', m.group()).upper()  # standardize name
            self.load_case = load_case
        else:
            print('WARNING: load case could not be found from path:', self.out_file)
            self.load_case = np.nan

    def read_file(self):
        with open(self.out_file, 'rU') as fh:
            out_lines = fh.readlines()
        out_lines = [line.strip() for line in out_lines
                     if line.strip() and not line.startswith('#')]
        self.out_lines = out_lines

    def read_ambient_data(self):
        lines = [line for line in self.out_lines if line.lower().startswith('carv')]
        try:
            car_v = float(lines[0].split(':')[1].strip('m/sCbar'))
        except:
            print('WARNING: car speed could not be identified', self.out_file)
            car_v = np.nan

        lines = [line for line in self.out_lines if line.lower().startswith('ambientt')]
        try:
            amb_t = float(lines[0].split(':')[1].strip('m/sCbar'))
        except:
            print('WARNING: ambient temperature could not be identified', self.out_file)
            amb_t = np.nan

        lines = [line for line in self.out_lines if line.lower().startswith('ambientp')]
        try:
            amb_p = float(lines[0].split(':')[1].strip('m/sCbar'))
        except:
            print('WARNING: ambient pressure could not be identified', self.out_file)
            amb_p = np.nan

        return car_v, amb_t, amb_p

    def read_hx_data(self, n_hx=4):
        '''collect heat exchanger data to a dataframe named df_hx'''
        for i, line in enumerate(self.out_lines):
            if line.startswith('HX'):
                break
        raw_data = [line.split() for line in self.out_lines[i+1:i+1+n_hx]]
        header = self.out_lines[i].split()
        hx_names = [row[0] for row in raw_data]
        # df = pd.DataFrame(data=data, columns=self.out_lines[i].split())
        multi_header = pd.MultiIndex.from_product([header[1:], hx_names], names=['var', 'hx'])
        multi_index = pd.MultiIndex.from_product([(self.load_case, ), (self.model_ver, )],
                                                 names=['loadcase', 'model'])
        data = np.array([row[1:] for row in raw_data]).astype(float)
        data = np.reshape(data.T, data.size)
        df = pd.DataFrame(data=[data], columns=multi_header, index=multi_index)
        self.df_hx = df

    def read_comp_temps(self):
        '''read temperature data for parts'''
        for i, line in enumerate(self.out_lines):
            if 'max wallt' in line.lower():
                break
        parts, temps = [], []
        for line in self.out_lines[i+1:]:
            try:
                part, temp = line.split(':')
                parts += [part.strip()]
                temps += [float(temp) - 273.15]  # convert to degC
            except:
                print('End of part temp section reached')
                break
        # multi_header = pd.MultiIndex.from_product([self.load_case, self.model_ver],
        #                                           names=['loadcase', 'model'])
        # df = pd.DataFrame(data=temps, columns=multi_header, index=parts)
        multi_index = pd.MultiIndex.from_product([parts, (self.load_case, )],
                                                 names=['part', 'loadcase'])
        df = pd.DataFrame(data=temps, columns=[self.model_ver], index=multi_index)
        print('check:\n', df)
        self.df_part_data = df


def plot_hx_data(df_hx):
    mon_vars = df_hx.columns.levels[0]
    hx_names = df_hx.columns.levels[1]
    loadcases = sorted(df_hx.index.levels[0], reverse=True)
    model_vers = df_hx.index.levels[1]
    fig, axes = plt.subplots(len(mon_vars), len(loadcases), sharex=True, sharey=False,
                             squeeze=False)
    for (i, var), (j, loadcase) in it.product(enumerate(mon_vars), enumerate(loadcases)):
        ax = axes[i, j]
        # print(i, j, var, loadcase)
        print(df_hx[var].loc[loadcase].T)
        df_hx[var].loc[loadcase].T.plot.bar(ax=ax, legend=False, title=loadcase)
        if j == 0:
            ax.set_ylabel(var)
    # handles = [ax.get_legend_handles_labels()[0] for ax in axes.flatten()]
    handles, labels = ax.get_legend_handles_labels()
    leg = plt.figlegend(handles, labels, fancybox=True, loc='upper right', bbox_to_anchor=(1.2, 1))
    fig.tight_layout()
    fig.savefig('hx_plots.png', bbox_extra_artists=(leg, ), bbox_inches='tight')


def plot_part_temps(df):
    model_vers = sorted(df.columns.unique(), reverse=True)
    parts = sorted(df.index.levels[0])
    #  barplot = df.plot.bar(y='loadcase', subplots=True)  # , layout=(2,2))
    fig, axes = plt.subplots(len(parts), 1, sharex=True, sharey=True,
                             squeeze=False)
    supt = fig.suptitle('Maximum part temperatures [$^\circ$C]', fontsize=12)
    for i, part in enumerate(parts):
        ax = axes[i, 0]
        print(df.loc[part])
        df.loc[part].plot.bar(ax=ax, legend=False)
        ax.set_ylabel(part)

    handles, labels = ax.get_legend_handles_labels()
    leg = plt.figlegend(handles, model_vers, fancybox=True, loc='upper right', bbox_to_anchor=(1.2, 1))
    fig.tight_layout()
    fig.savefig('part_plots.png', bbox_extra_artists=(leg, supt), bbox_inches='tight')

if __name__ == '__main__':
    out_files = find_out_files()
    print(out_files)
    hx_dframes, part_dframes = [], []
    for out_file in out_files:
        out = OutFile(out_file)
        hx_dframes += [out.df_hx]
        part_dframes += [out.df_part_data]

    df_hx = pd.concat(hx_dframes)  # , ignore_index=True)

    df_part = pd.concat(part_dframes, axis=1)
    print(df_hx)
    print(df_part)
    # sort out nans from df_part
    df_part.fillna(axis=1, method='backfill', inplace=True)
    dfs = []
    for col in df_part.columns.unique():
        dfs += [df_part[col].iloc[:, 0]]
    df_part = pd.concat(dfs, axis=1)
    print(df_part)

    # plot hx data
    plot_hx_data(df_hx)
    # barplot = df_hx['Tin[c]'].loc['TT30'].T.plot(kind='bar')
    # fig = barplot.get_figure()
    # fig.savefig('part_plots2.png')

    # plot temp data
    plot_part_temps(df_part)

    # write to excel
    writer = pd.ExcelWriter('cfd_results.xlsx')
    df_hx.to_excel(writer, sheet_name='COOLING')
    df_part.to_excel(writer, sheet_name='HP')
    writer.save()
