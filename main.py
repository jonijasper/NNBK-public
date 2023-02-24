"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html

"""
import random
import time

import numpy as np
import pandas as pd

from scipy import interpolate

from sklearn.neural_network import MLPRegressor
from sklearn import model_selection, metrics

from matplotlib import pyplot as plt


from setup import fileinfo, general


class TrainingData():
    """Training data model

    Args:
        name (string): Name to differentiate between different setups. 
            Should be unique since used in filenames when saving files. 
            Previous files with same name will be overwritten.
    
    Kwargs:
        These are optional arguments that can be used to modify the
        default settings.

        input_type (str): Choose type of the input data.
            Possible values are: 'F2', 'FL', 'HERA'.
            Defaults to 'F2'.

        parameters (list(string)): List of initial condition paramatrisations to use
            in training. Possible ic's are:  'mvg', 'mve', 'mvge', 'mv2g', 'gbw', 'atan', or 'all'.
            Defaults to  ['mvge'].

        saturation_scale (float): Saturation scale value N(r^2 = 1/Qs^2) = saturation_scale. 
            Used in saturation filter.  Use plot_saturation_scales() method to plot saturation scales.
            Defaults to 1-np.exp(-0.25).

        saturation_range (list): Range of saturation scale values that pass the saturation filter.
            Use plot_saturation_scales() method to plot saturation scales.
            Defaults to [0.1,1.0]. 

    """
    
    def __init__(self, 
                 name: str, 
                 input_type: str = "F2",
                 parameters: list|str = ['mvge'],
                 saturation_scale: float = general.SAT_SCALE,
                 saturation_range: list = [0.1,1.0]
                 ):
        
        self.name = name

        # check if valid input_type
        if input_type not in {'F2', 'FL', 'HERA'}:
            raise Exception(f"Invalid input type '{input_type}'")
        
        if 'all' in parameters:
            parameters = ['mvg', 'mve', 'mvge', 'mv2g', 'gbw', 'atan']

        self.settings = {"input_type": input_type, 
                         "parameters": parameters,
                         "saturation_scale": saturation_scale,
                         "saturation_range": saturation_range}
        
        self.filedict = self._make_filedict()
        self.training_files = None
        self.q_list = None
        self.x_list = None
        self.r_break = None

    def _saturation_filter(self, filelist):
        '''
        '''
        sat_scale = self.settings['saturation_scale']
        satrange = self.settings['saturation_range']
        sat_min = satrange[0]
        sat_max = satrange[1]

        filtered_files = []
        
        for file in filelist:
            if self.settings['input_type'] == 'HERA':
                Nrs, rs, sigmaym = rcs_reader(file)
            else:
                Nrs, rs, F2s = F2_reader(file)

            if sat_scale==None:
                SAT_Pass = True
            else: 
                r_sat = np.interp(sat_scale,Nrs,rs)
                Qs2 = 1/r_sat**2
                SAT_Pass = sat_min <= Qs2 <= sat_max
                
            if SAT_Pass:
                filtered_files.append(file)

        return filtered_files

    def _make_filedict(self):
        '''
        
        '''
        all_files = dict()
        for ic in self.settings['parameters']:
            # list all files
            files = [f"{fileinfo.F2}f2_{i}.dat" for i in fileinfo.filenros[ic]["range"]]

            # saturation filter
            filtered = self._saturation_filter(files)

            all_files[ic] = filtered

        return all_files

    def _remove_N_zeros(self,Nrs,rs,file):
        '''
        
        '''
        # choose method to deal with N=0 values:

        # --- skip whole files where N values nonloggable ---
        # runlogger(self.name,
        #             {"file skipped, N(r)=0 prevents log(N)":
        #                 f"\n{f2filename} ({bk_})\n{params_}"})
        # skipped_counter[icstr] += 1
        # continue


        # --- remove individual N(r)=0 points ---
        f2filename = file.split('/')[-1]
        bk_, icstr, params_ = get_ic_params(file)
        for _i,_N in enumerate(Nrs):
            if _N > 0:
                break
        runlogger(self.name,
                {f"{_i}/{len(Nrs)} N(r) values removed (N=0) from":
                        f"\n{f2filename} ({bk_})\n{params_}"})

        Nrs = Nrs[_i:]
        rs = rs[_i:]
        return Nrs, rs
    
    def _make_filelist(self,files_to_skip):
        '''
        
        '''
        all_files = []
        for ic,filelist in self.filedict.items():
            # skip testfiles
            testfiles_removed = [filename for filename in filelist
                    if filename not in files_to_skip]
        
            # equalize the number of files between ic's according to self.equalize
            if self.equalize and len(testfiles_removed)>self.equalize:
                random.seed(general.RAND_SEED)
                all_files.append(random.sample(testfiles_removed,self.equalize))
                print(f"{len(all_files[-1])} {ic} files (trimmed to eq)")
            else:
                all_files.append(testfiles_removed)
                print(f"{len(all_files[-1])} {ic} files")
    
        runlogger(self.name,
                    {f"{ic} files": len(all_files[-1])})
        
        return all_files

    def _remove_half_params():
        pass

    def _remove_half_F2s():
        pass
    
    def _input_interpolator():
        pass

    def create_training_data(self, files_to_skip: list):
        """ Creates input and output for network to use in training

        Args:
            files_to_skip (list): files that will not be added to training data
                so they can be used in testing.

        """
        self.training_files = self._make_filelist(files_to_skip)

        # check that test files are out
        if set.intersection(set(files_to_skip),set(self.training_files)):
            raise Exception(f"Testing files found in training files.")
        
        filecounter = 0
        skipped_counter = {'mvg': 0, 'mve': 0, 'gbw': 0, 'atan': 0, 
          'mv2g': 0, 'mvge': 0}

        train_input = []
        train_output = []

        for file in self.training_files:
            if self.hera:
                Nrs, rs, sigma_df = rcs_reader(file)
                F2s = sigma_df['Sigma'].tolist()

            else:
                Nrs, rs, F2s = F2_reader(file)

            if not self.r_break:
                for i, r in enumerate(rs):
                    if r >= self.settings['min_r']:
                        break
                self.r_break = i

            rs = rs[self.r_break:]
            Nrs = Nrs[self.r_break:]

            # test smallest r
            if rs[0] < self.settings['min_r']:
                raise Exception(f"smallest r={rs[0]} which is smaller than 'min_r':{self.settings['min_r']}")

            # deal with N(r)=0 cases, when taking log(N)
            if self.LOG_N and (0 in Nrs):
                Nrs,rs = self._remove_N_zeros(Nrs,rs,file)

            # add N(r)=1 points to large r 
            if self.settings['add_large_r']:
                add_r = np.logspace(2,3,21)[1:]
                add_Nr = np.ones(len(add_r))
                rs.extend(add_r)
                Nrs.extend(add_Nr)
                runlogger(self.name,
                            {f"added N(r)=1 points:":
                                f"{len(add_r)} from r={add_r[0]} to r={add_r[-1]}"})

            filecounter += 1
            
            # add input and output arrays
            if self.LOG_F:
                F2s = np.log(F2s)
            if self.LOG_N:
                Nrs = np.log(Nrs)

            if self.settings['two_r_input']:
                for _Nr,_r in zip(Nrs,rs):
                    train_input.append([_r,np.log(_r),*F2s])
                    train_output.append(_Nr)
            
            else:
                if self.LOG_R:
                    rs = np.log(rs)
            
                for _Nr,_r in zip(Nrs,rs):
                    train_input.append([_r,*F2s])
                    train_output.append(_Nr)


        runlogger(self.name,{"skipped/ics":""})
        runlogger(self.name, skipped_counter)


        runlogger(self.name,
            {f"{i} rs < min_r={self.settings['min_r']} removed": "",
            f"{filecounter} different ic in training data":""})

        print(f"\n{filecounter} different ic in training data")

        if self.hera:
            self.q_list = sigma_df['Q2'].tolist()
            self.x_list= sigma_df['x'].tolist()

        self.inputs = train_input
        self.outputs = train_output
    
    def create_test_data(self, file, testpoint_count):
        '''
        
        '''
        if self.settings['input_type'] == 'HERA':
            Nrs, rs, sigma_df = rcs_reader(file)
            F2s_orig = sigma_df['Sigma'].tolist()
        else:    
            Nrs, rs, F2s_orig = F2_reader(file)
    
        if not self.r_break:
            for i, r in enumerate(rs):
                if r >= self.settings['min_r']:
                    break
            self.r_break = i

        rs = rs[self.r_break:]
        Nrs = Nrs[self.r_break:]

        F2s = F2s_orig

        test_rs = np.logspace(np.log10(rs[0]),np.log10(rs[-1]),testpoint_count)
        Nrs_interpolator = interpolate.interp1d(rs,Nrs)
        test_Nrs = Nrs_interpolator(test_rs)

        if self.LOG_R:
            test_rs = np.log(test_rs)
        if self.LOG_F:
            F2s = np.log(F2s)
        if self.LOG_N:
            test_Nrs = np.log(test_Nrs)

        if self.ADD_LOGR:
            input = [[r,np.log(r),*F2s] for r in test_rs]
        else:
            input = [[r,*F2s] for r in test_rs]
        output = test_Nrs
        
        if self.hera:
            return input, output, F2s_orig, sigma_df
        else:
            return input, output, F2s_orig
    
    def plot_saturation_scale(self, satscale=None, satrange=None):
        if not satscale:
            satscale = self.settings['saturation_scale']
        if not satrange:
            satrange = self.settings['saturation_range']

        satmin = satrange[0]
        satmax = satrange[1]

        counts = dict()
        count = 0
        Qs = []

        for ic in self.settings['parameters']:
            for i in fileinfo.filenros[ic]["range"]:
                file = f"{fileinfo.F2}f2_{i}.dat" 
                if self.settings['input_type'] == 'HERA':
                    Nrs, rs, stuff = rcs_reader(file)
                else:
                    Nrs, rs, F2s = F2_reader(file)
                
                r_sat = np.interp(satscale,Nrs,rs)
                Qs.append(1/r_sat**2)

                if satmin <= Qs[-1] <= satmax:
                    count +=1
            indx = list(fileinfo.filenros[ic]["range"])
            plt.scatter(indx,Qs,label=ic, marker="+")
            counts[ic] = (count,len(indx))
            Qs = []
            count = 0

        print("number of files in each ic that would pass the filter")
        for key,value in counts.items():
            print(f"{key}: {value[0]}/{value[1]}")
        plt.legend()
        plt.axhline(y = satmin, color = 'r', linestyle = '--')
        plt.axhline(y = satmax, color = 'r', linestyle = '--')
        plt.suptitle(f"Saturation scale N$(r^2=1/Q^2_s)$ = {satscale:.2f}\n$Q_s^2$ cut $({satmin}...{satmax})$")
        plt.legend(loc=2, prop={'size': 8})
        plt.xlabel("filenro 'f2_i.dat'")
        plt.ylabel("$Q_s^2$ [GeV$^2$]")
        plt.yscale("log")
        plt.tight_layout()

        plt.savefig(f"{general.save_dir}{self.name}-saturation_scales.pdf")
        plt.show()


class Network():
    """Neural network

    Args:
        name (string): Name to differentiate between different setups. 
            Should be unique since used in filenames when saving files. 
            Previous files with same name will be overwritten.
        dummy (boolean): Create dummy network to compare scoring

    """
    
    def __init__(self, name: str, dummy=False, **kwargs):

        self.name = name
        self.regressor = MLPRegressor(**kwargs)

        if dummy:
            self.dummy_regr = MLPRegressor()

    def _prediction_uncertainty(self,
                                test_input, 
                                number_of_variations: int =100, 
                                standard_deviation: float =0.05,
                                hera_error: list =None):
        '''Shifts F2 values in input using gaussian dirstribution around the F2 values,
        makes predictions from variated inputs, and calculates standard deviation of
        the predictions for each r.

        Returns:
            pred_std, pred_mean, predlist
        '''
        gaussian_input = []
        pred_std = []
        predlist = []

        r_list = np.array(test_input).T[0]
        # TODO: two r input
        two_r_input = False
        if two_r_input:
            f2input = test_input[0][2:]
        else:
            f2input = test_input[0][1:]

        # TODO: make better check if F2 or log(F2)
        log_F2 = any(f2<0 for f2 in f2input)

        if log_F2:
            f2values = np.exp(f2input)
        else:
            f2values = f2input

        a = general.RANDOM_SEED+1
        b = a + number_of_variations
        for s in range(a, b):
            random.seed(s)
            gaussed_values = []
            if hera_error:
                for sigma,e in zip(f2values,hera_error):
                    gaussed_values.append(random.gauss(sigma,0.01*e*sigma))

            else:    
                for f2 in f2values:
                    gaussed_values.append(random.gauss(f2,standard_deviation*f2))

            if log_F2:
                if any(num_<=0 for num_ in gaussed_values):
                        raise Exception("STAND_DEV too large, non positive F2 values")
                gauss_in = np.log(gaussed_values)
            else:
                gauss_in = gaussed_values

            if two_r_input:
                gaussian_input = [[r,np.log(r), *gauss_in] for r in r_list]
            else: 
                gaussian_input = [[r, *gauss_in] for r in r_list]

            predictions = self.predict(gaussian_input)
            pred = predictions[1]

            # TODO: N>1 cut
            CUT_N = False
            LOG_N = False
            if CUT_N:
                if LOG_N:
                    linear_prediction = np.exp(pred)
                    linear_prediction[linear_prediction>1] = 1
                    pred = np.log(linear_prediction)
                else:
                    pred[pred>1] = 1

            predlist.append(pred)

        pred_std = np.std(predlist, axis=0)
        pred_mean = np.mean(predlist, axis=0)

        return pred_std, pred_mean, predlist

    def test(testdata):
        '''
        
        '''
        pass

    def train(self, training_data: TrainingData, train_frac: float):
        '''Trains the network
        
        Args:
            training_data (obj): training data instance used for training

            train_frac (float), Value from range [0,1]. Sets the fraction of data used in training, 
                rest is reserved for scoring
        
        '''
        input_ = training_data.inputs
        output_ = training_data.outputs

        if not (input_ and output_):
            raise Exception("create input and output from data")

        train_input, test_input, train_output, test_output = model_selection.train_test_split(
                    input_, output_, random_state=general.RANDOM_SEED,
                    train_size=train_frac)

        runlogger(f"runlogs/{training_data.name}-runinfo.txt", 
                    {"Datapoints splitted. Training points (rest for scoretest)":
                        f"{len(train_input)}/{len(input_)}",
                    "Number of inputs": 
                        len(train_input[0])})

        if self.dummy_regr:
            self.dummy_regr.fit(train_input,train_output)

        self.regressor.fit(train_input, train_output)

        self._test_input = test_input
        self._test_output = test_output

    def score(self, test_input: list =None, test_output: list =None, weights: bool =False):
        '''
        
        '''
        dummy_pred, test_pred = self.predict(test_input)
        
        if not (test_input and test_output):
            print("using default test split for scoring")
            test_input = self._test_input
            test_output = self._test_output

        if self.LOG_N:
            dummy_pred = np.exp(dummy_pred)
            test_pred = np.exp(test_pred)
            test_output = np.exp(test_output)
        
        if weights:
            w = 1/np.array(test_pred)
        else:
            w = None

        rmse_score = metrics.mean_squared_error(test_output,test_pred,
                        sample_weight=w,squared=False)
        
        R2_score = self.regressor.score(test_input,test_output)

        if self.dummy_regr:
            rmse_dummy = metrics.mean_squared_error(test_output,dummy_pred,
                        sample_weight=w,squared=False)
            R2_dummy = self.dummy_regr.score(test_input,test_output)
        else:
            rmse_dummy = -1
            R2_dummy = -1
            
        print(f"{self.name} scores for testgroup (dummy model):\n"\
            f"  R2: {R2_score:.3f} ({R2_dummy:.3f})\n"\
            f"  RMSE: {rmse_score:.1e} ({rmse_dummy:.1e})\n")

        runlogger(self.name,
                    {"R2 score (dummy model)": f"{R2_score} ({R2_dummy})",
                    f"RMSE (dummy model)": f"{rmse_score} ({rmse_dummy})",})

    def plot_testgroup_predictions(self, test_output, preds):
        '''
        
        '''
        # fig0,ax0 = plt.subplots()
        # ax0.scatter(test_output,preds[1], s=4, c=setup.colors[2], label="model")
        # if self.dummy_regr:
        #     ax0.scatter(test_output,preds[0], s=4, c=setup.colors[3],label="dummy")
        
        # ax0.plot([0,1],[0,1], 'k--', alpha=0.9, label="x=y")

        # ax0.set_xlim(0,1)
        # ax0.set_ylim(0,1.5)
        # ax0.set_ylabel("network prediction")
        # ax0.set_xlabel("target value")

        # ax0.legend(**setup.leg_kwargs)
        # plt.tight_layout()
        # fig0.savefig(f"{setup.save_dir}{self.name}-testgroup_preds.pdf")
        # # plt.show()
        # plt.close(fig0)
        pass

    def predict(self, input):
        '''
        
        '''
        if self.dummy_regr:
            dummy_pred = self.dummy_regr.predict(input)
        else:
            dummy_pred = -1

        test_pred = self.nn.predict(input)

        return dummy_pred, test_pred

    def loss_plot():
        pass



def get_ic_params(file):
    with open(file) as f:
        BKfile = f.read().split('#')[1].split()[0].split('/')[-1].split('.')[0]
        if '_' in BKfile:
            icstr, nrstr = BKfile.split('_')
        
        else:
            icstr = ""
            nrstr = ""
            for char in BKfile:
                if char.isdigit():
                    nrstr+=char
                if char.isalpha():
                    icstr+=char
    infofile=f"../NNBK/data/datafiles/BKevo/constant-coupling/_info_{icstr}.txt"
    row_nr = int(nrstr) -1
    with open(infofile) as f:
        ic_params = f.read().split('\n')[row_nr].split(': ')[-1]
    asd = ic_params.split(', ')
    if icstr=="mv2g":
        Q1 = float(asd[0].split('=')[1])
        Q2 = float(asd[1].split('=')[1])
        g1 = float(asd[2].split('=')[1])
        g2 = float(asd[3].split('=')[1])
        params = {"$Q_{s0,1}^2$": Q1,
                    "$\gamma_1$": g1,
                    "$Q_{s0,2}^2$": Q2,
                    "$\gamma_2$":g2}
    else:
        Qs02 = float(asd[0].split('=')[1])
        params = {"$Q_{s0}^2$": Qs02}
        
        for x in asd[1:]:
            y= x.split('=')
            if y[0] == "gamma":
                params["$\gamma$"] = y[1]
            if y[0] == "ec":
                params["$e_c$"] = y[1]
    
    params_str = ""
    for key, value in params.items():
        if key == "$\gamma_1$":
            params_str += f"{key} = {value},\n        "
        else:
            params_str += f"{key} = {value}, "
    return BKfile, icstr, params_str[:-2]

def runlogger(name: str, contents: dict) -> None:
    '''  '''
    with open(f"{general.save_dir}runinfo-{name}.txt", "a") as f:
        for key, value in contents.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

def F2_reader(filename: str) -> list:
    '''reads .dat files made with BKtoF2-calculator'''

    with open(filename) as f:
        content = f.read().split("###")

    content = content[1:]   # gets rid of the stuff at the beginning
    content = [i.split() for i in content] # cleans up lines
    thedata = []
    for i in content:
        x = list(map(float, i))
        thedata.append(x)

    Nrs, rs, F2s = thedata

    return Nrs, rs, F2s

def rcs_reader(filename: str) -> tuple:
    '''reads .dat files made with BKtoCS-calculator'''

    with open(filename) as f:
        stuff, initN, rs, sigmaym = f.read().split("###")

    initN = initN.split() # cleans up lines
    initN = list(map(float,initN))
    rs = rs.split() # cleans up lines
    rs = list(map(float,rs))

    sigmaym = sigmaym.split('\n')[1:]
    titles = sigmaym[0].split()
    sigma_dict = dict()

    sigmalist = []
    for row in sigmaym[1:]:
        temp = row.split()
        sigmalist.append(list(map(float,temp)))
    sigmalist = np.array(sigmalist)
    for title_,list_ in zip(titles,sigmalist.T):
        sigma_dict[title_] = list_
    
    sigma_df = pd.DataFrame.from_dict(sigma_dict)

    return initN, rs, sigma_df

def time_it(func):
    '''This function shows the execution time of 
    the function object passed'''
    def wrap_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        exec_time = time.time() - start_time
        if exec_time >= 60*60:
            exec_time = f"{exec_time/60/60:.2f} hours"
        elif exec_time >= 60:
            exec_time = f"{exec_time/60:.2f} minutes"
        else:
            exec_time = f"{exec_time:.2f} seconds"
            
        print(f'Function {func.__name__!r} executed in {exec_time}')
        return result
    return wrap_func

@time_it
def main(datamodels: list|object, testfiles: list|str):
    # train networks for different data models
    trained_networks = []
    for data in datamodels:
        data.create_training_data(files_to_skip=testfiles)
        nn = Network()
        nn.train(data,general.train_frac)
        nn.score(data.testdata)
        trained_networks.append(nn)
        
    # run tests for trained networks
    for nn in trained_networks:
        nn.test(testfiles)


if __name__ =="__main__":
    data1 = TrainingData("data_all", parameters = ['all'])
    data1.plot_saturation_scale()
    # datamodels = [data1]
    # testfiles = []
    # main(datamodels, testfiles)