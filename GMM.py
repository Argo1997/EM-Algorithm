import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class gmm:

    def __init__(self, k, X):
        self.k = k
        self.fnk = []
        self.m, self.n = X.shape
        self.weights = np.full(shape=self.k, fill_value=1 / self.k)
        self.average = np.full(shape=X.shape, fill_value=1 / self.k)
        random = np.random.randint(low=0, high=self.m, size=self.k)

        # The following comments can be uncommented to initialize random mean and covariance matrices

        # self.mean = [X[idx, :] for idx in random]

        self.mean = [[-0.49282153, -12.37705277], [10.94702438, 18.59835075], [19.85757915, 17.34078882],
                     [-12.56857912, 4.85978422], [-3.92027833, -0.16528674]]

        # self.var = [(self.covariance(X, self.mean[idx])) for idx in range(self.k)]

        self.var = [[1.0290593, 0.84134949],
                    [0.84134949, 1.50818392]], [[0.9716829, 0.79945857],
                                                [0.79945857, 1.43994984]], [[0.86698681, 0.56596332],
                                                                            [0.56596332, 1.09824126]], [
                       [0.94633081, 0.69274814],
                       [0.69274814, 1.38106501]], [[0.95003936, 0.72007888],
                                                   [0.72007888, 1.37181161]]

        self.likelihood = np.zeros([self.m, self.k], dtype=float)

    def covariance(self, X, mean):
        cov = []

        for i in range(self.n):
            temp = []
            for j in range(self.n):
                x = [float(X[idx, i]) - float(mean[i]) for idx in range(self.m)]
                y = [float(X[idx, j]) - float(mean[j]) for idx in range(self.m)]
                temp.append(np.sum([x[idx] * y[idx] for idx in range(self.m)]) / self.m)
            cov.append(temp)
        return cov

    def expectation(self, X):

        # Updating Probabilities
        self.fnk = []
        self.likelihood = np.zeros([self.m, self.k], dtype=float)
        for i in range(self.k):
            mu = self.mean[i]
            var = self.var[i]
            distribution = multivariate_normal(mu, var)
            self.likelihood[:, i] = distribution.pdf(X)

        for i in range(self.m):
            temp = []
            for j in range(self.k):
                temp.append(
                    (self.weights[j] * self.likelihood[i, j]) / np.sum(np.multiply(self.weights, self.likelihood[i, :]),
                                                                       axis=0))
            self.fnk.append(temp)
        self.fnk = np.array(self.fnk)

    def maximization(self, X):

        Nk = [np.sum(self.fnk[:, i]) for i in range(self.k)]

        # Updating Class Weights
        self.weights = np.multiply((1 / self.m), [Nk[i] for i in range(self.k)])

        # Updating Mean
        temp = np.zeros([self.k, self.n], dtype=float)

        for j in range(self.k):
            temp[j] = np.sum([(self.fnk[i, j] * X[i, :]) for i in range(self.m)], axis=0)
        self.mean = [(temp[i] / Nk[i]) for i in range(self.k)]

        # Updating Covariance Matrix
        temp = []

        for j in range(self.k):
            container = np.zeros([self.n, self.n], dtype=float)
            outer = [np.outer((X[i, :] - self.mean[j]), (X[i, :] - self.mean[j])) for i in range(self.m)]
            bin = [np.multiply(outer[i], self.fnk[i, j]) for i in range(self.m)]

            for id in range(self.m):
                container = container + bin[id]
            temp.append(container)
        self.var = [np.multiply(1 / Nk[i], temp[i]) for i in range(self.k)]

    def learn(self, X, iter):
        for i in range(iter):
            print("Iteration ", i, "/100")
            self.expectation(X)
            self.maximization(X)

    def predict(self, X):
        self.expectation(X)
        return np.argmax(self.fnk, axis=1)

data = pd.read_csv("gmm_data.txt", delimiter=" ")
X = np.array(data)
k = 5
model = gmm(k, X)
model.learn(X, 100)
labels = model.predict(X)


# Plotting the ellipse
def plot_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    # PCA
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        ang = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        wi, hi = 2 * np.sqrt(s)
    else:
        ang = 0
        wi, hi = 2 * np.sqrt(covariance)
    # Draw the Ellipse
    for nsig in range(1, 4):
        ellipse = Ellipse(position, nsig * wi, nsig * hi, ang, **kwargs)
        ax.add_artist(ellipse)
    ax.set_xlim(-20, 25)
    ax.set_ylim(-20, 25)

w_factor = 0.4 / model.weights.max()

for i in range(5):
    plot_ellipse(model.mean[i], model.var[i], alpha=model.weights[i] * w_factor)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

