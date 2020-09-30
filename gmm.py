import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class dataloader:

    def __init__(self):

        self.parse_data()

    def parse_data(self):

        # Extract data
        inputdata = hdf5storage.read(filename='raw_mixed_full.h5')[0][0]
        self.data = np.asarray(inputdata['data'])
        self.label = np.array(inputdata['label'])
        self.pid = np.array(inputdata['PatientId'])
        self.inv = np.array(inputdata['inv'])

        self.data, self.inv, self.pid, self.label = self.format_data(self.data, self.inv, self.pid, self.label)
        self.data, self.inv, self.pid, self.label = self.by_inv(self.data, self.inv, self.pid, self.label, 0.6, 'gt')

        self.data = np.reshape(self.data, (-1, 1))

    @staticmethod
    def format_data(data, inv, pid, label):
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

        return X, I, P, Y

    @staticmethod
    def by_label(data, inv, pid, label, label_value):
        # returns only cores of X with specified label
        # if all of data, inv, and label are provided, returns all three for entries with specified label

        mask = label == label_value

        return data[mask], inv[mask], pid[mask], label[mask]

    @staticmethod
    def by_inv(data, inv, pid, label, inv_value, cond):
        # returns only rows of X with involvement satisfying condition
        # if all of data, inv, and label are provided, returns all three for entries with the condition

        if (cond == 'gt'):
            mask = inv >= inv_value
        else:
            mask = inv <= inv_value

        return data[mask], inv[mask], pid[mask], label[mask]

class gmm:

    def __init__(self):

        self.val_fold = val_fold
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self):

        # Boilerplate
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # Init network
        self.dl = dataloader(val_fold=self.val_fold)
        self.build_model()
        self.classifier.summary()

        # Prepare callbacks
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.000)
        filepath = "model-cls-epoch-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=False, period=10)

        # Fit
        # self.history = self.classifier.fit(x=self.dl.data_train, y=self.dl.label_train, validation_data=[self.dl.data_val, self.dl.label_val], nb_epoch=self.epochs, batch_size=self.batch_size, verbose=1).history

    def build_model(self):

        num_layers = len(self.n_filters)
        assert(len(self.kernel_size) == num_layers and len(self.strides) == num_layers)

        # Input layer
        x1 = Input(shape=(50,), name='modelinput')

        # Conv layers
        x = Reshape((50, 1))(x1)
        for i in range(len(self.n_filters)):
            x = Conv1D(self.n_filters[i], self.kernel_size[i], strides=self.strides[i], padding='same', activation='linear', kernel_regularizer=l1_l2(0.05, 0.10), bias_regularizer=l1_l2(0.10, 0.05), activity_regularizer=l1_l2(0.05, 0.10))(x)
            x = LeakyReLU()(x)
            x = MaxPool1D(self.pool_size[i])(x)

        # Fully connected layer
        features = Flatten()(x)
        features = Dense(250, activation='relu', kernel_regularizer=l1_l2(0.05, 0.10), bias_regularizer=l1(0.5), activity_regularizer=l1_l2(0.05, 0.10))(features)
        output = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(0.25, 0.50), bias_regularizer=l1(0.8))(features)

        # Define model
        self.classifier = Model(inputs=x1, outputs=output, name='classifier')

        # Specificity and sensitivity metric functions
        def sensitivity(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            return true_positives / (possible_positives + K.epsilon())

        def specificity(y_true, y_pred):
            true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
            possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
            return true_negatives / (possible_negatives + K.epsilon())

        # Compile model
        self.classifier.compile(optimizer=self.optimizer, loss=binary_crossentropy, metrics=['accuracy', sensitivity, specificity])
