from os import listdir
from os.path import join
import numpy as np
import sklearn.manifold
from unidecode import unidecode
import matplotlib.pyplot as plt


def terke(text, n):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """
    text = unidecode(text)
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("--", "-")
    text = text.lower()
    terke = {}
    for i in range(len(text) - n + 1):
        str = text[i:i + n]
        if str in terke:
            terke[str] += 1
        else:
            terke[str] = 1
    return terke


def read_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("jeziki"):
        if fn.lower().endswith(".txt"):
            with open(join("jeziki", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds


def cosine_dist(d1, d2):
    """
    Vrne kosinusno razdaljo med slovarjema terk d1 in d2.
    """
    l1 = []
    l2 = []
    for key in d1:
        if key in d2:
            l1.append(d1[key])
            l2.append(d2[key])

    if not l1:
        return 1

    n1 = np.linalg.norm(l1)
    n2 = np.linalg.norm(l2)
    dist = np.dot(l1, l2) / (n1 * n2)

    return 1 - dist


def prepare_data_matrix(data_dict):
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf (NOT the complete tf-idf) measure.
    """

    # creating a dictionary, which instead of frequencies of terms
    # stores their idf-s in order
    idf_dict = {}
    triplets_list = []
    for lang in data_dict:
        for term in data_dict[lang]:
            idf_dict[term] = idf(term, data_dict)

    for triplet in idf_dict:
        triplets_list.append((triplet, idf_dict[triplet]))

    # sort by idf
    triplets_list.sort(key=lambda x: x[1])
    triplets_list = triplets_list[:100]

    # compute frequencies of 100 most common triplets
    # for each document in the directiory
    X = np.zeros((len(data_dict), 100))

    for i, lang in enumerate(data_dict):
        for j in range(100):
            term = triplets_list[j][0]
            if term in data_dict[lang]:
                value = data_dict[lang][term]
                X[i][j] = value

    sums = X.sum(axis=1)
    X = X / sums[:, np.newaxis]

    # get language id-s from text file names
    text_names = list(data_dict)
    languages = []
    for text_name in text_names:
        #languages.append(text_name.split('_')[1].split('.')[0])
        languages.append(text_name.split('-')[1].split('.')[0])

    return X, languages


def idf(t, corpus):
    # compute idf
    n = len(corpus)
    count = 0
    for lang in corpus:
        if t in corpus[lang]:
            count += corpus[lang][t]

    return np.log(n / count)


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """
    # centering the values
    X = X - np.mean(X, axis=0)

    # covariance matrix
    C = np.cov(X, rowvar=False)

    # initialization of a random vector
    v = np.random.rand(C.shape[1])

    # compute eigenvector of largest eigenvalue
    error = 1
    while error > 0.000001:
        w = np.dot(C, v)
        w = w / np.linalg.norm(w)
        error = np.linalg.norm(v - w)
        v = w

    # get eigenvalue
    val = np.dot(v, np.dot(C, v))

    return v, val


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    # first component
    eig_vect1, eig_val1 = power_iteration(X)

    new_X = X - X @ np.outer(eig_vect1, eig_vect1)

    # second component
    eig_vect2, eig_val2 = power_iteration(new_X)

    return np.array([eig_vect1, eig_vect2]), np.array([eig_val1, eig_val2])


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    return (vecs @ (X - np.mean(X, axis=0)).T).T


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    return np.sum(eigenvalues) / total_variance(X)


def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """

    lds = read_data(3)

    X, languages = prepare_data_matrix(lds)

    vects, vals = power_iteration_two_components(X)

    Y = project_to_eigenvectors(X, vects)

    expl_var = explained_variance_ratio(X, vects, vals)

    # plot
    fig, tile = plt.subplots()
    tile.scatter(Y[:, 0], Y[:, 1])
    tile.set_title("PCA Explained Variance: " + str(expl_var))
    for i, label in enumerate(languages):
        tile.annotate(label, Y[i])

    plt.show()


def dissimilarity(lds):
    '''compute dissimilarity matrix of
    dictionary lds'''

    dismat = np.zeros((len(lds), len(lds)))

    for i, lang1 in enumerate(lds):
        for j, lang2 in enumerate(lds):
            if i != j:
                dismat[i][j] = cosine_dist(lds[lang1], lds[lang2])

    return dismat


def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    lds = read_data(3)

    X, languages = prepare_data_matrix(lds)

    X_new = sklearn.manifold.MDS(n_components=2, dissimilarity="precomputed").fit_transform(dissimilarity(lds))

    # plot
    fig, tile = plt.subplots()
    tile.scatter(X_new[:, 0], X_new[:, 1])
    tile.set_title("MDS")
    for i, label in enumerate(languages):
        tile.annotate(label, X_new[i])

    plt.show()


def plot_TSNE():

    lds = read_data(3)

    X, languages = prepare_data_matrix(lds)

    X_new = sklearn.manifold.TSNE(n_components=2, metric="precomputed", learning_rate="auto",
                                  init="random").fit_transform(dissimilarity(lds))

    # plot
    fig, tile = plt.subplots()
    tile.set_title("t-SNE")
    tile.scatter(X_new[:, 0], X_new[:, 1])
    for i, label in enumerate(languages):
        tile.annotate(label, X_new[i])

    plt.show()


if __name__ == "__main__":
    # each method needs a few seconds to compute the output
    # once the figure of a method is shown, close it in order to run the next method
    plot_MDS()
    plot_PCA()
    plot_TSNE()
