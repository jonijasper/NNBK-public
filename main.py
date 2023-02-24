import random
import time

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn import model_selection, metrics

from setup import fileinfo, general


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

    def test(testdata):
        pass

    def train(self, training_data, train_frac):
        '''Trains the network and calculates performance score 
        (set in setup.setup) by using a fraction trainfrac of 
        training data to train and rest are set aside for scoring
        
        Args:
            training_data (obj): training data instance used for training

        :train_frac: float, fraction of data used in training, rest is used for
                            cross validation scoring
        
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

    def score(self, test_input=None, test_output=None, weights=False):
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

        runlogger(f"{self.name}-runinfo.txt",
                    {"R2 score (dummy model)": f"{R2_score} ({R2_dummy})",
                    f"RMSE (dummy model)": f"{rmse_score} ({rmse_dummy})",})

    def plot_testgroup_predictions(self, test_output, preds):
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
        if self.dummy_regr:
            dummy_pred = self.dummy_regr.predict(input)
        else:
            dummy_pred = -1

        test_pred = self.nn.predict(input)

        return dummy_pred, test_pred


class TrainingData():
    """Training data model

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Args:
        name (string): Name to differentiate between different setups. 
            Should be unique since used in filenames when saving files. 
            Previous files with same name will be overwritten.
    
    Optional kwargs:
        These are optional arguments that can be used to modify the
        default settings.

        hera (boolean): Use hera data instead of mock data.
        parameters (list(string)): List of initial condition paramaters to use
        in training.

    """
    def __init__(self, 
                 name: str, 
                 *,
                 hera: bool = False,
                 parameters: list|str = ['mvge'],
                 ) -> None:
        

        self.name = name
        self.settings = {"hera": hera, 
                         "parameters": parameters}
        

        self.filedict = self._make_filedict()
        self.training_files = None
        self.q_list = None
        self.x_list = None

    def _saturation_filter(self, filelist):
        '''
        
        '''
        sat_scale = self.settings['saturation_scale']
        sat_min = self.settings['sat_min']
        sat_max = self.settings['sat_max']
        filtered_files = []
        
        for file in filelist:
            if self.settings['hera']:
                Nrs, rs, sigmaym = sigma_reader(file)
            else:
                Nrs, rs, F2s = F2reader(file)

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
            files = [f"{fileinfo.F2}_{i}.dat" for i in fileinfo.filenros[ic]]

            # saturation filter
            filtered = self.saturation_filter(files)

            all_files[ic] = filtered

        return all_files

    def _remove_N_zeros(self,Nrs,rs,file):
        '''
        
        '''
        # choose method to deal with N=0 values:

        # --- skip whole files where N values nonloggable ---
        '''
        # runlogger(f"{self.name}-runinfo.txt",
        #             {"file skipped, N(r)=0 prevents log(N)":
        #                 f"\n{f2filename} ({bk_})\n{params_}"})
        # skipped_counter[icstr] += 1
        # continue
        '''

        # --- remove individual N(r)=0 points ---
        f2filename = file.split('/')[-1]
        bk_, icstr, params_ = funcs.get_ic_params(file)
        for _i,_N in enumerate(Nrs):
            if _N > 0:
                break
        runlogger(f"{self.name}-runinfo.txt",
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
    
        runlogger(f"{self.name}-runinfo.txt",
                    {f"{ic} files": len(all_files[-1])})
        
        return all_files

    def create_input_output(self, files_to_skip: list):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

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
        r_break=None

        for file in self.training_files:
            if self.hera:
                Nrs, rs, sigma_df = sigma_reader(file)
                F2s = sigma_df['Sigma'].tolist()

            else:
                Nrs, rs, F2s = F2reader(file)

            if not r_break:
                for i, r in enumerate(rs):
                    if r >= self.settings['min_r']:
                        break
                r_break = i

            rs = rs[r_break:]
            Nrs = Nrs[r_break:]

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
                runlogger(f"{self.name}-runinfo.txt",
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


        runlogger(f"{self.name}-runinfo.txt",{"skipped/ics":""})
        runlogger(f"{self.name}-runinfo.txt", skipped_counter)


        runlogger(f"{self.name}-runinfo.txt",
            {f"{i} rs < min_r={self.settings['min_r']} removed": "",
            f"{filecounter} different ic in training data":""})

        print(f"\n{filecounter} different ic in training data")

        if self.hera:
            self.q_list = sigma_df['Q2'].tolist()
            self.x_list= sigma_df['x'].tolist()

        self.inputs = train_input
        self.outputs = train_output


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

def runlogger(file, contents) -> None:
    ''' f"runinfo-{self.name}.txt" '''
    with open(general.save_dir + file, "a") as f:
        for key, value in contents.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

def F2reader(filename: str) -> list:
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

def sigma_reader(filename: str) -> tuple:
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
        data.create_input_output(files_to_skip=testfiles)
        nn = Network()
        nn.train(data,general.train_frac)
        nn.score(data.testdata)
        trained_networks.append(nn)
        
    # run tests for trained networks
    for nn in trained_networks:
        nn.test(testfiles)


if __name__ =="__main__":
    datamodels = [TrainingData("data1")]
    testfiles = []

    main(datamodels, testfiles)