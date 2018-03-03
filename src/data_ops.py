from sklearn import datasets
import numpy as np


# convert 2d image to 1d vector
#def vec__(x): return x.reshape(x.shape[:1] + (-1,))
vec__ = lambda x: x.reshape(x.shape[:1] + (-1,))


def convert_one_hot(labels, dim):
    n = len(labels)
    targets = np.zeros((n, dim))
    targets[np.arange(n), labels] = 1
    return targets


def shuffle__(data, labels):
    rand_indx = np.arange(len(data))
    np.random.shuffle(rand_indx)
    return data[rand_indx], labels[rand_indx]


def filter_2__(data, labels, pos_class=2, neg_class=9):
    data_indx = (labels==pos_class) + (labels==neg_class)
    data, labels = data[data_indx], labels[data_indx]
    labels[labels==pos_class] = 1
    labels[labels==neg_class] = 0
    return data, labels

    
def split2__(data, labels, percentage1):
    cut = int(len(labels)*percentage1)
    return data[:cut], labels[:cut], data[cut:], labels[cut:]


def make_classification(n_samples, labeled_percentage):
    data, labels = datasets.make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                                n_informative=2, n_clusters_per_class=1)
    data, labels = shuffle__(data, labels)
    return split2__(data, labels, labeled_percentage)


def gen_half_circle__(n_samples, sigma):
    x = np.linspace(0, np.pi, n_samples)
    sin_x = np.sin(x) + np.random.randn(n_samples)*sigma
    xn = x + np.random.randn(n_samples)*sigma
    return xn, sin_x

    
def make_half_moons(n_training, n_test, noise=None):
    data, labels = datasets.make_moons(n_samples=n_training + n_test, shuffle=True, noise=noise, random_state=None)
    #train_data, train_labels = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=noise, random_state=None)
    #test_data, test_labels = datasets.make_moons(n_samples=n_test, shuffle=True, noise=noise, random_state=None)
    return data[:n_training], labels[:n_training], data[n_training:], labels[n_training:]

   
def make_half_moons_ssl(n_unlabeled, n_labeled, n_test):
    train_data, train_labels, test_data, test_labels = make_half_moons(n_unlabeled + n_labeled, n_test)
    return train_data[:n_unlabeled], train_labels[:n_unlabeled], train_data[n_unlabeled:], train_labels[n_unlabeled:], test_data, test_labels


def make_half_circles(n_samples, labeled_percentage, separation=.5, sigma=.1, shuffle=True, signed_labels=False):
    n_samples_2 = int(n_samples/2)
    x1, y1 = gen_half_circle__(n_samples_2, sigma)
    x2, y2 = gen_half_circle__(n_samples-n_samples_2, sigma)
    x2 += (max(x2)-min(x2))*separation

    c1 = np.column_stack((x1, y1))
    l1 = np.zeros(len(x1))
    c2 = np.column_stack((x2, y2))
    l2 = np.ones(len(x2))

    data = np.concatenate((c1, c2))
    labels = np.concatenate((l1, l2))
    if signed_labels:
        labels = labels*2 - 1
    if shuffle:
        data, labels = shuffle__(data, labels)
    return split2__(data, labels, labeled_percentage)


def load_mnist_digits(src='tensorflow', shuffle=False):
    if src=='tensorflow':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('../data/', one_hot=False)
        train_data, train_labels = mnist.train.images, mnist.train.labels
        test_data, test_labels = mnist.test.images, mnist.test.labels
        valid_data, valid_labels = mnist.validation.images, mnist.validation.labels

        train_data, train_labels = np.vstack([train_data, valid_data]),\
                                   np.concatenate([train_labels, valid_labels])
    elif src=='sklearn':
        digits = datasets.load_digits()
        div = np.amax(digits.data)
        train_data, train_labels = digits.data/div, digits.target
        test_data, test_labels = digits.data/div, digits.target
    else:
        raise('Undefined mnist source.')

    if shuffle:
        train_data, train_labels = shuffle__(train_data, train_labels)
        test_data, test_labels = shuffle__(test_data, test_labels)
    return train_data, train_labels, test_data, test_labels

    
def load_mnist_digits_ssl(labeled_percentage=None,labeled_count=None):
    train_data, train_labels, test_data, test_labels = load_mnist_digits()
    if labeled_count==None:
        lab_data, lab_labels, unlab_data, unlab_labels = split2__(train_data, train_labels, labeled_percentage)
    else:
        cut = labeled_count
        data, labels = train_data, train_labels
        lab_data, lab_labels, unlab_data, unlab_labels = data[:cut], labels[:cut], data[cut:], labels[cut:]
    return lab_data, lab_labels, unlab_data, unlab_labels, test_data, test_labels


def load_mnist_digits_2(src):
    train_data, train_labels, test_data, test_labels = load_mnist_digits(src)
    train_data, train_labels = shuffle__(*filter_2__(train_data, train_labels))
    test_data, test_labels = shuffle__(*filter_2__(test_data, test_labels))
    return train_data, train_labels, test_data, test_labels


def load_mnist_char__(file='data/notMNIST.npz', shuffle=True):
    data_pack = np.load(file)
    data, labels = data_pack['images'], data_pack['labels']
    rand_indx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(rand_indx)
    data, labels = data[rand_indx]/255., labels[rand_indx]
    return data, labels


def load_mnist_char():
    data, labels = load_mnist_char__()
    train_data, train_labels = vec__(data[:15000]), labels[:15000]
    test_data, test_labels = vec__(data[15000:]), labels[15000:]
    return train_data, train_labels, test_data, test_labels

    
def load_mnist_char_2():
    data, labels = load_mnist_char__()
    data, labels = filter_shuffle_2__(data, labels)
    train_data, train_labels = vec__(data[:3500]), labels[:3500]
    test_data, test_labels = vec__(data[3500:]), labels[3500:]
    return train_data, train_labels, test_data, test_labels

    
import matplotlib.pyplot as plt


def vec2img(vec):
    pixels = np.array(vec)#*255.0, dtype='uint8')
    dim = int(np.sqrt(len(vec)))
    pixels = pixels.reshape((dim, dim))
    return pixels

def plot_mnist_setup_title_(path, index, label, index_first):
    arg1, arg2 = (index, label) if index_first is True else (label, index)
    title = '{arg1}_{arg2}'.format(arg1=arg1, arg2=arg2)
    file = '%s/%s.jpg'%(path,title)
    #plt.title(title)
    return file

        
def plot_mnist(data, labels, path, index_first=False):
    for vec, label, index in zip(data, labels, range(len(labels))):
        pixels = vec2img(vec)
        file = plot_mnist_setup_title_(path, index, label, index_first)
        plt.imshow(pixels, cmap='gray')
        #plt.show()
        plt.savefig(file)
        

def plot_mnist_pairs(data1, data2, labels, path, index_first=False):
    for vec1, vec2, label, index in zip(data1, data2, labels, range(len(labels))):
        pixels1 = vec2img(vec1)
        pixels2 = vec2img(vec2)
        # Plot
        file = plot_mnist_setup_title_(path, index, label, index_first)
        plt.subplot(121)
        plt.imshow(pixels1, cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(pixels2, cmap='gray')
        plt.axis('off')
        #plt.show()
        plt.savefig(file)
        
        
def load_mnist_char_depricated():
    data = np.load("notMNIST.npz")
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]


def load_data_depricated1(n, a, b, c, d, e):
    x = np.random.randn(n)*a
    y = np.random.randn(n)*b
    return np.column_stack((x + d, c*(x + y) + e))


def load_data_depricated2(n_samples):
    x1 = gen_data(int(n_samples/2), 2, 3, 1, 10, 2)
    x2 = gen_data(n_samples - len(x1), 2, 6, 0.5, 1, 2)

    data = np.concatenate((x1, x2))
    labels = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))))
    randInd = np.arange(len(data))
    np.random.shuffle(randInd)
    data, labels = data[randInd], labels[randInd]
    return data, labels
    
    
import numpy

class DataSet():
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
            # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


import logging
from general import *

class SemiSupervisedTrainer:
    def __init__(self, batch_size_L, batch_size_U, X_L, T_L, X_U, X_T, T_T, sim_id, init_file, dump_freq=600, logger_timeout=30):
        self.sim_id = sim_id
        self.batch_size_L = batch_size_L
        self.batch_size_U = batch_size_U
        self.dump_freq = dump_freq

        self.pipe_L = DataSet(X_L, T_L)
        self.pipe_U = DataSet(X_U, X_U)
        self.test_data = (X_T, T_T)
        
        history = load_pik_try('../cache/%s__history_last.pik'%self.sim_id)
        is_history = init_file and history
        logging.getLogger('Trainer').info('Initializing %s.'%('empty history' if is_history is None else 'history from dumped file'))
        self.history = history if is_history else []
        self.logger_timeout = logger_timeout
        
    def dump_state(self, model, sess):
        dump_pik('../cache/%s__history_%s.pik'%(self.sim_id,time_id()), self.history)
        dump_pik('../cache/%s__history_last.pik'%self.sim_id, self.history)
        model.dump_model(sess)
        
    def train(self, sess, n_epochs, optimizer):
        model = optimizer.model
        logger = logging.getLogger('Trainer')
        logger.info("Starting training model:%s."%model.sim_id)
        last_epoch, tt1, tt2 = -1, TimeTracker(), TimeTracker()

        while self.pipe_L.epochs_completed < n_epochs:
            X_L, T_L = self.pipe_L.next_batch(self.batch_size_L)
            X_U, _ = self.pipe_U.next_batch(self.batch_size_U)
            optimizer.optimize(sess, X_L, T_L, X_U)
            
            if tt1.elapsed(self.dump_freq, True):
                self.dump_state(model, sess)
                
            timeout, epoch_completed = False, False
            if self.pipe_L.epochs_completed > last_epoch:
                last_epoch = self.pipe_L.epochs_completed
                epoch_completed = True
                
            if tt2.elapsed(self.logger_timeout, True):
                timeout = True
                
            if timeout or epoch_completed:
                X_T, T_T = self.test_data
                stats = optimizer.gen_stats(sess, X_L, T_L, X_U, X_T, T_T)
                line = ', '.join('%s: %.3f'%(field,stat) for field, stat in zip(optimizer.stat_fields, stats))
                logger.debug("Epoch: %d, %s"%(last_epoch, line))
                if epoch_completed: self.history.append(stats)
                timeout, epoch_completed = False, False

        self.dump_state(model, sess)
        logger.info("Finished training.")
        return {'history':self.history, 'fields':optimizer.stat_fields}
    