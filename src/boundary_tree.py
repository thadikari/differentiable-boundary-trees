import numpy as np
from scipy import spatial


class Node:
    def __init__(self, y, l):
        self.value = np.array(y)
        self.label = l
        self.child_nodes = []
        self.child_values = np.zeros(shape=(0, len(y)))

    def add(self, y, l):
        self.child_nodes.append(Node(y, l))
        self.child_values = np.vstack([self.child_values, y])

    def size(self):
        return len(self.child_nodes)


class Tree:
    def __init__(self, dim, k):
        self.root = None
        self.size = 0
        self.k = k

    def train(self, y, l):
        if self.root is None:
            self.root = Node(y, l)
            self.size = 1
            return True
        else:
            v = self.query(y)
            if v.label == l:
                return False

            self.size += 1
            v.add(y, l)
            return True

    def query(self, y):
        v = self.root
        while 1:

            if v.size() == 0:
                return v

            child_dists = spatial.distance.cdist(
                v.child_values, np.array([y]), 'euclidean')
            child_ind = np.argmin(child_dists)
            child_min = child_dists[child_ind]
            parent_dist = spatial.distance.euclidean(v.value, y)

            if (v.size() < self.k) & (parent_dist < child_min):
                return v
            else:
                v = v.child_nodes[child_ind]


class Forest:
    def __init__(self, dim, n, k):
        self.dim = dim
        self.n = n
        self.k = k
        self.trees = []

    def train(self, y, l):
        if len(self.trees) < self.n:
            self.trees.append(Tree(self.dim, self.k))

        for T in self.trees:
            T.train(y, l)


__dim = lambda arg: len(arg[0])


class Set:
    def __init__(self, dim, values=None, labels=None):
        self.dim = dim
        self.values = np.zeros(shape=(0, dim)) if values is None else values
        self.labels = [] if labels is None else labels

    @classmethod
    def from_values(cls, values, labels):
        return cls(len(values[0]), values, labels)

    def __add(self, y, l):
        self.values = np.vstack([self.values, y])
        self.labels.append(l)

    @property
    def size(self): return len(self.labels)

    def train(self, y, l):
        if self.values.size == 0:
            self.__add(y, l)
            return True
        else:
            value, label = self.query(y)
            if label == l:
                return False
            else:
                self.__add(y, l)
                return True

    def query(self, y):
        dists = spatial.distance.cdist(
            self.values, np.array([y]), 'euclidean')
        ind = np.argmin(dists)
        value = self.values[ind]
        label = self.labels[ind]
        return value, label


def build_boundary_set_ex(data, labels):
    result = []
    b_set = Set(len(data[0]))
    for y, l in zip(data, labels):
        result.append(b_set.train(y, l))
    return b_set, result


def build_boundary_set(data, labels):
    return build_boundary_set_ex(data, labels)[0]


def __draw(p, plt):
    if p.child_nodes:
        for v in p.child_nodes:
            plt.plot((v.value[0], p.value[0]),
                     (v.value[1], p.value[1]), 'g-')
            __draw(v, plt)


def simulate_tree(k, data, labels, plt):
    t = Tree(__dim(data), k)
    for y, l in zip(data, labels):
        t.train(y, l)

    plt.scatter(data[:, 0], data[:, 1], marker='.', s=20, c=labels)
    __draw(t.root, plt)
    plt.axis('equal')


def simulate_set(data, labels, plt):
    s = Set(__dim(data))
    for y, l in zip(data, labels):
        s.train(y, l)

    plt.scatter(data[:, 0], data[:, 1], marker='.', s=5, edgecolor='none', c=labels)
    print(len(s.labels), len(s.values))
    plt.scatter(s.values[:, 0], s.values[:, 1], marker='s', s=30, c=s.labels, edgecolor='1')
    plt.axis('equal')


def simulate_forest(n, k, data, labels, plt):
    f = Forest(__dim(data), n, k)
    s = Set(len(data[0]))
    for y, l in zip(data, labels):
        f.train(y, l)
        s.train(y, l)

    for T, i in zip(f.trees, range(10)):
        plt.subplot(1, 3, i+1)
        plt.scatter(data[:, 0], data[:, 1], marker='.', s=20, c=labels)
        plt.scatter(s.values[:, 0], s.values[:, 1], marker='s', s=70,
                    c=s.labels, edgecolor='1')
        __draw(T.root, plt)
        plt.axis('equal')


def run_tests():
    from sklearn.datasets import make_moons, make_classification
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    #from loading import make_data

    np.random.seed(676)
    #data, labels = make_data(n_samples=10000)

    data, labels = make_moons(n_samples=10000, shuffle=True, noise=None, random_state=None) # make_circles

    #data, labels = make_moons(n_samples=1000, shuffle=True, noise=None, random_state=None)
    #data, labels = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    #simulate_forest(3, 3, data, labels, plt)
    #simulate_tree(3, data, labels, plt)

    num = 10
    start = time.time()
    print("starting")
    for _ in range(num):
        s = Set(len(data[0]))
        for y, l in zip(data, labels):
            s.train(y, l)
        #print(s.labels)
    end = time.time()
    print((end - start)/num)

    simulate_set(data, labels, plt)
    plt.show()


if __name__ == "__main__":
    run_tests()
