import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import tensorflow as tf
import keras
import gc

class dataloader:

    def __init__(self, dataset='testing', norm_method='min_max_per_core', val_fold=1, crop=None, inv_thresh=0.4, custom_data=None, format='Hierarchical', verbose=0):
        # Arguments:
        # dataset = 'balanced' or 'extended' or 'mixed' (or 'testing' for bogus data)
        # norm method = 'L2', 'min_max', 'max', or None
        # fold = between 1 and 5, indicates how validation set is split from training
        # crop = tuple of (start, finish) or None
        # inv_thresh = between 0 and 1 value of involvement we're interested in; can be used in conjunction with ~custom~
        # custom = 'benign', 'cancer', 'high_inv'
        # format = Squashed for all the signals to be concatenated into one massive array, Hierarchical for the signals to be grouped based on patient and then core
        # verbose = 0 for silent, 1 for summary, 2 for summary + progress

        # Initialize
        if (verbose == 2):
            print("Initializing...")
        self.data_train = None
        self.label_train = None
        self.inv_train = None
        self.pid_train = None
        self.data_val = None
        self.label_val = None
        self.inv_val = None
        self.pid_val = None
        self.data_test = None
        self.label_test = None
        self.inv_test = None
        self.pid_test = None

        # Load data
        if (verbose == 2):
            print("Loading data...")
        if (dataset == 'balanced'):
            self.parse_BK('BK_RF_P1_90.mat', val_fold)
        elif (dataset == 'extended'):
            self.parse_BK('BK_RF_P1_90-ext.mat', val_fold)
        elif (dataset == 'mixed'):
            self.parse_BK('BK_RF_P1_90-wcb.mat', val_fold)
        elif (dataset == 'testing'):
            self.load_bogus(val_fold)
        else:
            raise ValueError('Unexpected value for parameter ~dataset~ (expected ~balanced~ or ~extended~ or ~mixed~)')

        # Extract dataset stats
        nc_train_benign = len([label for label in self.label_train if label == 0])
        nc_train_cancer = len([label for label in self.label_train if label == 1])
        nc_train_hi_inv_cancer = len([inv for inv in self.inv_train if inv >= inv_thresh])
        nc_val_benign = len([label for label in self.label_val if label == 0])
        nc_val_cancer = len([label for label in self.label_val if label == 1])
        nc_val_hi_inv_cancer = len([inv for inv in self.inv_val if inv >= inv_thresh])
        nc_test_benign = len([label for label in self.label_test if label == 0])
        nc_test_cancer = len([label for label in self.label_test if label == 1])
        nc_test_hi_inv_cancer = len([inv for inv in self.inv_test if inv >= inv_thresh])

        # Crop data
        if (crop is not None):
            if (verbose == 2):
                print("Cropping data...")
            self.data_train = self.crop_data(self.data_train, crop)
            self.data_val = self.crop_data(self.data_val, crop)
            self.data_test = self.crop_data(self.data_test, crop)

        # Apply normalization
        if (verbose == 2):
            print("Applying normalization...")
        self.normalize(norm_method)

        # Get custom data sets
        if (custom_data == 'benign'):
            self.data_train, self.inv_train, self.pid_train, self.label_train = self.byLabel(self.data_train, self.label_train, 0, self.inv_train, self.pid_train)
            self.data_val, self.inv_val, self.pid_val, self.label_val = self.byLabel(self.data_val, self.label_val, 0, self.inv_val, self.pid_val)
            self.data_test, self.inv_test, self.pid_test, self.label_test = self.byLabel(self.data_test, self.label_test, 0, self.inv_test, self.pid_test)
            nc_train_cancer = 0
            nc_train_hi_inv_cancer = 0
            nc_val_cancer = 0
            nc_val_hi_inv_cancer = 0
            nc_test_cancer = 0
            nc_test_hi_inv_cancer = 0
        elif (custom_data == 'cancer'):
            self.data_train, self.inv_train, self.pid_train, self.label_train = self.byLabel(self.data_train, self.label_train, 1, self.inv_train, self.pid_train)
            self.data_val, self.inv_val, self.pid_val, self.label_val = self.byLabel(self.data_val, self.label_val, 1, self.inv_val, self.pid_val)
            self.data_test, self.inv_test, self.pid_test, self.label_test = self.byLabel(self.data_test, self.label_test, 1, self.inv_test, self.pid_test)
            nc_train_benign = 0
            nc_val_benign = 0
            nc_test_benign = 0
        elif (custom_data == 'hi_inv'):
            self.data_train, self.inv_train, self.pid_train, self.label_train = self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.pid_train, self.label_train)
            self.data_val, self.inv_val, self.pid_val, self.label_val = self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.pid_val, self.label_val)
            self.data_test, self.inv_test, self.pid_test, self.label_test = self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.pid_test, self.label_test)
            nc_train_benign = 0
            nc_train_cancer = nc_train_hi_inv_cancer
            nc_val_benign = 0
            nc_val_cancer = nc_val_hi_inv_cancer
            nc_test_benign = 0
            nc_test_cancer = nc_test_hi_inv_cancer
        elif (custom_data == 'benign+hi_inv'):
            dT, iT, pT, lT = self.byLabel(self.data_train, self.label_train, 0, self.inv_train, self.pid_train)
            dT = np.append(dT, self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.pid_train, self.label_train)[0], axis=0)
            iT = np.append(iT, self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.pid_train, self.label_train)[1])
            pT = np.append(pT, self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.pid_train, self.label_train)[2])
            lT = np.append(lT, self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.pid_train, self.label_train)[3])
            self.data_train = dT
            self.inv_train = iT
            self.pid_train = pT
            self.label_train = lT
            dV, iV, pV, lV = self.byLabel(self.data_val, self.label_val, 0, self.inv_val, self.pid_val)
            dV = np.append(dV, self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.pid_val, self.label_val)[0], axis=0)
            iV = np.append(iV, self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.pid_val, self.label_val)[1])
            pV = np.append(pV, self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.pid_val, self.label_val)[2])
            lV = np.append(lV, self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.pid_val, self.label_val)[3])
            self.data_val = dV
            self.inv_val = iV
            self.pid_val = pV
            self.label_val = lV
            dS, iS, pS, lS = self.byLabel(self.data_test, self.label_test, 0, self.inv_test, self.pid_test)
            dS = np.append(dS, self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.pid_test, self.label_test)[0], axis=0)
            iS = np.append(iS, self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.pid_test, self.label_test)[1])
            pS = np.append(pS, self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.pid_test, self.label_test)[2])
            lS = np.append(lS, self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.pid_test, self.label_test)[3])
            self.data_test = dS
            self.inv_test = iS
            self.pid_test = pS
            self.label_test = lS
            nc_train_cancer = nc_train_hi_inv_cancer
            nc_val_cancer = nc_val_hi_inv_cancer
            nc_test_cancer = nc_test_hi_inv_cancer
        else:
            pass

        # Print a summary
        if (verbose != 0):

            print("------------")

            if (dataset == 'balanced'):
                print("Dataset:                BK_RF_P1_90.mat")
            else:
                print("Dataset:                BK_RF_P1_90-ext.mat")

            print("Validation fold:        " + str(val_fold))

            print(" --> Training:          " + str(nc_train_benign+nc_train_cancer) + " total cores (" + str(self.countSignals(self.data_train)) + " data points)")
            print("                        " + str(nc_train_benign) + " benign cores (" + str(self.countSignals(self.byLabel(self.data_train, self.label_train, 0))) +  " data points)")
            print("                        " + str(nc_train_cancer) + " cancer cores (" + str(self.countSignals(self.byLabel(self.data_train, self.label_train, 1))) +  " data points)")
            print("                        " + str(nc_train_hi_inv_cancer) + " high involvement (>=" + str(inv_thresh) + ") cancer cores (" + str(self.countSignals(self.byInv(self.data_train, self.inv_train, inv_thresh, cond='gt'))) +  " data points)")

            print(" --> Validation:        " + str(nc_val_benign+nc_val_cancer) + " total cores (" + str(self.countSignals(self.data_val)) + " data points)")
            print("                        " + str(nc_val_benign) + " benign cores (" + str(self.countSignals(self.byLabel(self.data_val, self.label_val, 0))) +  " data points)")
            print("                        " + str(nc_val_cancer) + " cancer cores (" + str(self.countSignals(self.byLabel(self.data_val, self.label_val, 1))) +  " data points)")
            print("                        " + str(nc_val_hi_inv_cancer) + " high involvement (>=" + str(inv_thresh) + ") cancer cores (" + str(self.countSignals(self.byInv(self.data_val, self.inv_val, inv_thresh, cond='gt'))) +  " data points)")

            print(" --> Testing:           " + str(nc_test_benign+nc_test_cancer) + " total cores (" + str(self.countSignals(self.data_test)) + " data points)")
            print("                        " + str(nc_test_benign) + " benign cores (" + str(self.countSignals(self.byLabel(self.data_test, self.label_test, 0))) +  " data points)")
            print("                        " + str(nc_test_cancer) + " cancer cores (" + str(self.countSignals(self.byLabel(self.data_test, self.label_test, 1))) +  " data points)")
            print("                        " + str(nc_test_hi_inv_cancer) + " high involvement (>=" + str(inv_thresh) + ") cancer cores (" + str(self.countSignals(self.byInv(self.data_test, self.inv_test, inv_thresh, cond='gt'))) +  " data points)")

            print("Normalization method:   " + str(norm_method))

            if (crop is not None):
                print("Time-series cropping:   Frames " + str(crop[0]) + " to " + str(crop[1]))
            else:
                print("Time-series cropping:   None")

            print("------------")

        # Format data
        if (verbose == 2):
            print("Formatting data...")
        if (format == 'Squashed'):
            self.data_train, self.inv_train, self.pid_train, self.label_train = self.format_data_sq(self.data_train, self.inv_train, self.pid_train, self.label_train)
            self.data_val, self.inv_val, self.pid_val, self.label_val = self.format_data_sq(self.data_val, self.inv_val, self.pid_val, self.label_val)
            self.data_test, self.inv_test, self.pid_test, self.label_test = self.format_data_sq(self.data_test, self.inv_test, self.pid_test, self.label_test)
        elif (format == 'Hierarchical'):
            self.data_train, self.inv_train, self.pid_train, self.label_train = self.format_data_hr(self.data_train, self.inv_train, self.pid_train, self.label_train)
            self.data_val, self.inv_val, self.pid_val, self.label_val = self.format_data_hr(self.data_val, self.inv_val, self.pid_val, self.label_val)
            self.data_test, self.inv_test, self.pid_test, self.label_test = self.format_data_hr(self.data_test, self.inv_test, self.pid_test, self.label_test)
        else:
            raise ValueError('Unexpected value for parameter ~format~ (expected ~Squashed~ or ~Hierarchical~)')

    def parse_BK(self, dataset, fold):
        # Parses matlab datasets and populates training, validation, and test sets

        # Extract data
        inputdata = hdf5storage.loadmat(dataset)
        data = inputdata['data'][0]
        label = inputdata['label'][0]
        patient_id = inputdata['PatientId'][0]
        inv = inputdata['inv'][0]

        # Allocate train vs test
        test_id = [5, 6, 18, 45, 37, 30, 12, 14, 24, 46, 52, 58, 79, 71]
        test_idx = [True if pid in test_id else False for pid in patient_id]
        train_idx = [not tr for tr in test_idx]

        # Allocate train vs val
        if (fold == 1):
            fold1_id = [3, 4, 8, 21, 25, 29, 64, 80]
            train_idx = [False if pid in fold1_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold1_id else False for pid in patient_id]
        elif (fold == 2):
            fold2_id = [7, 11, 26, 40, 47, 57, 72, 87]
            train_idx = [False if pid in fold2_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold2_id else False for pid in patient_id]
        elif (fold == 3):
            fold3_id = [13, 19, 27, 59, 60, 65, 81, 85]
            train_idx = [False if pid in fold3_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold3_id else False for pid in patient_id]
        elif (fold == 4):
            fold4_id = [22, 23, 38, 39, 48, 68, 74, 82]
            train_idx = [False if pid in fold4_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold4_id else False for pid in patient_id]
        elif (fold == 5):
            fold5_id = [2, 10, 28, 42, 66, 70, 76, 89, 90]
            train_idx = [False if pid in fold5_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold5_id else False for pid in patient_id]
        else:
            raise ValueError('Unexpected value for parameter ~val_fold~ (expected 1,2,...,5)')

        # Separate train vs test set
        self.data_train = data[train_idx]
        self.label_train = label[train_idx]
        self.inv_train = inv[train_idx]
        self.pid_train = patient_id[train_idx]
        self.data_val = data[val_idx]
        self.label_val = label[val_idx]
        self.inv_val = inv[val_idx]
        self.pid_val = patient_id[val_idx]
        self.data_test = data[test_idx]
        self.label_test = label[test_idx]
        self.inv_test = inv[test_idx]
        self.pid_test = patient_id[test_idx]

    def load_bogus(self, fold):
        # Parses bogus numpy data

        # Extract data
        fn = 'bogus_data_fold_' + str(fold) + '.npy'

        # Parse data
        try:
            self.data_train, self.label_train, self.inv_train, self.pid_train, self.data_val, self.label_val, self.inv_val, self.pid_val, self.data_test, self.label_test, self.inv_test, self.pid_test = np.load(fn, allow_pickle=True)
        except:
            self.data_train, self.label_train, self.inv_train, self.pid_train, self.data_val, self.label_val, self.inv_val, self.pid_val, self.data_test, self.label_test, self.inv_test, self.pid_test = np.load('./.~datasets/' + fn, allow_pickle=True)

    def normalize(self, norm_method):

        if (norm_method == 'L2'):
            self.data_train = self.L2_norm(self.data_train)
            self.data_val = self.L2_norm(self.data_val)
            self.data_test = self.L2_norm(self.data_test)
        elif (norm_method == 'min_max'):
            self.data_train = self.min_max_norm(self.data_train)
            self.data_val = self.min_max_norm(self.data_val)
            self.data_test = self.min_max_norm(self.data_test)
        elif (norm_method == 'min_max_per_core'):
            self.data_train = self.min_max_per_core_norm(self.data_train)
            self.data_val = self.min_max_per_core_norm(self.data_val)
            self.data_test = self.min_max_per_core_norm(self.data_test)
        elif (norm_method == 'max'):
            self.data_train = self.max_norm(self.data_train)
            self.data_val = self.max_norm(self.data_val)
            self.data_test = self.max_norm(self.data_test)
        else:
            pass

    @staticmethod
    def L2_norm(data):
        # Performs L2 norm on data
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            mean_fea = np.mean(data[i], axis=0, keepdims=True) + 1e-6
            std_fea = np.std(data[i], axis=0, keepdims=True) + 1e-6
            out.append(np.divide(data[i] - mean_fea, std_fea))

        return out

    @staticmethod
    def min_max_norm(data):
        # Performs min-max norm on data
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            scaler = MinMaxScaler(feature_range=(0,1))
            scaler = scaler.fit(data[i].T)
            normed = scaler.transform(data[i].T)
            out.append(np.transpose(normed))

        return out

    @staticmethod
    def min_max_per_core_norm(data):
        # Performs min-max norm on each core (instead of each data point)
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            minimum = np.min(data[i])
            maximum = np.max(data[i])
            out.append((data[i] - minimum) / (maximum - minimum))

        return out

    @staticmethod
    def max_norm(data):
        # Performs max norm on data (i.e. divide data by max amplitude)
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            max_fea = np.max(data[i])
            out.append(np.divide(data[i], max_fea))

        return out

    @staticmethod
    def format_data_sq(data, inv, pid, label):
        # Outputs:
        # X (ndarray): total_data_points by num_timesteps
        # I (ndarray): total_data_points by 1
        # P (ndarray): total_data_points by 1
        # Y (ndarray): total_data_points by 1

        if (data is None or inv is None or pid is None or label is None):
            return None, None, None

        X = np.concatenate(data, axis=0)
        I = np.array([])
        for i in range(len(inv)):
            I = np.append(I, inv[i]*np.ones(data[i].shape[0]))
        P = np.array([])
        for i in range(len(pid)):
            P = np.append(P, pid[i]*np.ones(data[i].shape[0]))
        Y = np.array([])
        for i in range(len(label)):
            Y = np.append(Y, label[i]*np.ones(data[i].shape[0]))

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, I, P, Y

    @staticmethod
    def format_data_hr(data, inv, pid, label):
        # Outputs:
        # X (ndarray): num_patients by num_cores by num_signals by num_timesteps
        # I (ndarray): num_patients by num_cores by num_signals by num_timesteps
        # P (ndarray): num_patients by num_cores by num_signals by num_timesteps
        # Y (ndarray): num_patients by num_cores by num_signals by num_timesteps

        if (data is None or inv is None or pid is None or label is None):
            return None, None, None

        X = []
        I = []
        P = []
        Y = []
        for i in range(len(pid)):
            if (pid[i] in P):
                idx = P.index(pid[i])
                X[idx].append(np.reshape(data[i], (data[i].shape[0], data[i].shape[1], 1)))
                I[idx].append(inv[i])
                Y[idx].append(label[i])
            else:
                P.append(pid[i])
                X.append([np.reshape(data[i], (data[i].shape[0], data[i].shape[1], 1))])
                I.append([inv[i]])
                Y.append([label[i]])

        return X, I, P, Y

    @staticmethod
    def crop_data(data, crop):

        if (crop is None or data is None):
            return None

        # Check that cropping fits timeseries boundaries
        num_timestamps = data[0].shape[1]
        if (crop[0] < 0 or crop[1] > num_timestamps):
            raise ValueError('Cropping tuple out of range')

        out = []

        # Iterate through and crop
        for i in range(len(data)):
            out.append(data[i][:,crop[0]:crop[1]])

        return out

    @staticmethod
    def byLabel(X, Y, label, I=None, P=None):
        # returns only cores of X with specified label
        # if all of data, inv, and label are provided, returns all three for entries with specified label

        if (X is None or Y is None):
            return None

        Xout = []
        for i in range(len(Y)):
            if (Y[i] == label):
                Xout.append(X[i])

        if (I is None or P is None):
            return Xout
        else:
            Iout = []
            Pout = []
            Yout = []
            for i in range(len(Y)):
                if (Y[i] == label):
                    Iout.append(I[i])
                    Pout.append(P[i])
                    Yout.append(Y[i])
            return Xout, Iout, Pout, Yout

    @staticmethod
    def byInv(X, I, inv, cond, P=None, Y=None):
        # returns only rows of X with involvement satisfying condition
        # if all of data, inv, and label are provided, returns all three for entries with the condition

        if (X is None or I is None):
            return None

        Xout = []
        for i in range(len(I)):
            if (cond == 'gt' and I[i] >= inv):
                Xout.append(X[i])
            if (cond == 'lt' and I[i] <= inv):
                Xout.append(X[i])

        if (P is None or Y is None):
            return Xout
        else:
            Iout = []
            Pout = []
            Yout = []
            for i in range(len(I)):
                if (cond == 'gt' and I[i] >= inv):
                    Iout.append(I[i])
                    Pout.append(P[i])
                    Yout.append(Y[i])
                if (cond == 'lt' and I[i] <= inv):
                    Iout.append(I[i])
                    Pout.append(P[i])
                    Yout.append(Y[i])
            return Xout, Iout, Pout, Yout

    @staticmethod
    def countSignals(X):
        # Counts total number of signals in an array of cores

        count = 0
        for arr in X:
            count += arr.shape[0]

        return count


class visualizer:

    def __init__(self, model, history, dataset, norm_method, val_fold, crop, inv_thresh):

        # Copy over model
        self.model = model
        self.history = history
        self.dataset = dataset
        self.norm_method = norm_method
        self.val_fold = val_fold
        self.crop = crop
        self.inv_thresh = inv_thresh

    def training_curve(self):
        # Plot training curve

        print("Plotting training curve...")
        plt.close()
        plt.plot(self.history['val_loss'], label='validation')
        plt.plot(self.history['loss'], label='training')
        plt.legend()
        plt.savefig('training-curve.png')

    def benign_cancer_examples(self, num_examples=5):
        # Plot input-output examples for benign and cancer

        print("Generating data for plotting examples")
        benigns = dataloader(dataset=self.dataset,
                             norm_method=self.norm_method,
                             val_fold=self.val_fold,
                             crop=self.crop,
                             inv_thresh=self.inv_thresh,
                             custom_data='benign',
                             format='Squashed',
                             verbose=0).data_test
        cancers = dataloader(dataset=self.dataset,
                             norm_method=self.norm_method,
                             val_fold=self.val_fold,
                             crop=self.crop,
                             inv_thresh=self.inv_thresh,
                             custom_data='hi_inv',
                             format='Squashed',
                             verbose=0).data_test

        print("Plotting input-output examples for benign...")
        for i in range(num_examples):
            sample = np.reshape(benigns[np.random.randint(len(benigns)),:,0], (1, -1, 1))
            out_ex = np.reshape(self.model.predict(sample), (-1, 1))
            in_ex = np.reshape(sample, (-1, 1))
            loss = self.model.evaluate(x=sample, y=sample, batch_size=None, verbose=0)
            plt.close()
            plt.plot(out_ex, label='output')
            plt.plot(in_ex, label='input')
            plt.suptitle(str(loss))
            plt.legend()
            fname = 'input-output-benign-' + str(i) + '.png'
            plt.savefig(fname)

        print("Plotting input-output examples for cancer...")
        for i in range(num_examples):
            sample = np.reshape(cancers[np.random.randint(len(cancers)),:,0], (1, -1, 1))
            out_ex = np.reshape(self.model.predict(sample), (-1, 1))
            in_ex = np.reshape(sample, (-1, 1))
            loss = self.model.evaluate(x=sample, y=sample, batch_size=None, verbose=0)
            plt.close()
            plt.plot(out_ex, label='output')
            plt.plot(in_ex, label='input')
            plt.suptitle(str(loss))
            plt.legend()
            fname = 'input-output-cancer-' + str(i) + '.png'
            plt.savefig(fname)

        # Garbage collection
        del benigns
        del cancers
        gc.collect()

    def error_distribution(self, hist_range=None):
        # Plot error distributions

        print("Generating data for plotting examples")
        testd = dataloader(dataset=self.dataset,
                           norm_method=self.norm_method,
                           val_fold=self.val_fold,
                           crop=self.crop,
                           inv_thresh=self.inv_thresh,
                           custom_data=None,
                           format='Hierarchical',
                           verbose=0)

        testX = testd.data_test
        testY = testd.label_test
        testI = testd.inv_test
        testP = testd.pid_test

        big_benign = []
        big_lowinv = []
        big_inv02 = []
        big_inv04 = []
        big_inv06 = []
        big_inv08 = []

        for pid in range(len(testX)):
            benign_loss_vec = []
            lowinv_loss_vec = []
            inv02_loss_vec = []
            inv04_loss_vec = []
            inv06_loss_vec = []
            inv08_loss_vec = []
            plt.close()
            print("Plotting error distribution for patient ID " + str(pid) + " of " + str(len(testX)))
            for core in range(len(testX[pid])):
                core_loss_vec = tf.keras.losses.mean_squared_error(np.reshape(testX[pid][core], (testX[pid][core].shape[0], testX[pid][core].shape[1])),
                                                                   np.reshape(self.model.predict(testX[pid][core], batch_size=None), (testX[pid][core].shape[0], testX[pid][core].shape[1])))
                if (testY[pid][core] == 0):
                    benign_loss_vec.extend(core_loss_vec)
                if (testI[pid][core] < 0.2 and testY[pid][core] == 1):
                    lowinv_loss_vec.extend(core_loss_vec)
                if (testI[pid][core] >= 0.2):
                    inv02_loss_vec.extend(core_loss_vec)
                if (testI[pid][core] >= 0.4):
                    inv04_loss_vec.extend(core_loss_vec)
                if (testI[pid][core] >= 0.6):
                    inv06_loss_vec.extend(core_loss_vec)
                if (testI[pid][core] >= 0.8):
                    inv08_loss_vec.extend(core_loss_vec)
            sns.distplot(lowinv_loss_vec, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='Low Inv')
            sns.distplot(inv02_loss_vec, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.2 Inv')
            sns.distplot(inv04_loss_vec, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.4 Inv')
            sns.distplot(inv06_loss_vec, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.6 Inv')
            sns.distplot(inv08_loss_vec, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.8 Inv')
            sns.distplot(benign_loss_vec, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='Benign')
            print("LoInv: " + str(len(lowinv_loss_vec)))
            print("Inv02: " + str(len(inv02_loss_vec)))
            print("Inv04: " + str(len(inv04_loss_vec)))
            print("Inv06: " + str(len(inv06_loss_vec)))
            print("Inv08: " + str(len(inv08_loss_vec)))
            print("Bengn: " + str(len(benign_loss_vec)))
            plt.suptitle("Patient ID: " + str(pid) + " # Cores: " + str(len(testX[pid])))
            plt.xlabel("Mean Sqr Error")
            plt.legend()
            plt.savefig('benign-cancer-loss-distrib-' + str(pid) + '.png')
            big_benign.extend(benign_loss_vec)
            big_lowinv.extend(lowinv_loss_vec)
            big_inv02.extend(inv02_loss_vec)
            big_inv04.extend(inv04_loss_vec)
            big_inv06.extend(inv06_loss_vec)
            big_inv08.extend(inv08_loss_vec)

        plt.close()
        sns.distplot(big_benign, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='Benign')
        sns.distplot(big_lowinv, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='Low Inv')
        sns.distplot(big_inv02, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.2 Inv')
        sns.distplot(big_inv04, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.4 Inv')
        sns.distplot(big_inv06, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.6 Inv')
        sns.distplot(big_inv08, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3, 'clip': hist_range}, label='>0.8 Inv')
        plt.suptitle("Totalled loss distributions")
        plt.xlabel("Mean Sqr Error")
        plt.legend()
        plt.savefig('total-benign-cancer-loss-distrib.png')

    def generate_histograms(self, hist_range=None):

        print("Generating data for plotting examples")
        testd = dataloader(dataset=self.dataset,
                           norm_method=self.norm_method,
                           val_fold=self.val_fold,
                           crop=self.crop,
                           inv_thresh=self.inv_thresh,
                           custom_data=None,
                           format='Hierarchical',
                           verbose=0)

        testX = testd.data_test
        testY = testd.label_test
        testI = testd.inv_test
        testP = testd.pid_test

        hist_data = []
        hist_label = []
        hist_inv = []
        hist_pid = []

        for pid in range(len(testX)):
            print("Creating histogram for patient ID " + str(pid) + " of " + str(len(testX)))
            for core in range(len(testX[pid])):
                core_loss_vec = tf.keras.losses.mean_squared_error(np.reshape(testX[pid][core], (testX[pid][core].shape[0], testX[pid][core].shape[1])),
                                                                   np.reshape(self.model.predict(testX[pid][core], batch_size=None), (testX[pid][core].shape[0], testX[pid][core].shape[1])))
                hist_data.append(np.histogram(core_loss_vec, bins=50, range=hist_range, density=True))
                hist_label.append(testY[pid][core])
                hist_inv.append(testI[pid][core])
                hist_pid.append(testP[pid])

        save_dict = {'data': hist_data, 'label': hist_label, 'inv': hist_inv, 'PatientId': hist_pid}
        hdf5storage.write(save_dict)



