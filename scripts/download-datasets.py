import h5py
import numpy
import os
import random
import sys

from urllib.request import urlopen
from urllib.request import urlretrieve

# ADAPTED FROM https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/datasets.py

def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = 'http://ann-benchmarks.com/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            DATASETS[which](hdf5_fn)
    hdf5_f = h5py.File(hdf5_fn, 'r')

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_f.attrs['dimension']) if 'dimension' in hdf5_f.attrs else len(hdf5_f['train'][0])

    return hdf5_f, dimension


def get_dataset_fn(dataset):
    if not os.path.exists('data'):
        os.mkdir('data')
    return os.path.join('data', '%s.hdf5' % dataset)


def write_output(train, test, fn, distance, point_type='float', count=100):
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
    n = 0
    f = h5py.File(fn, 'w')
    f.attrs['type'] = 'dense'
    f.attrs['distance'] = distance
    f.attrs['dimension'] = len(train[0])
    f.attrs['point_type'] = point_type
    print('train size: %9d * %4d' % train.shape)
    print('test size:  %9d * %4d' % test.shape)
    f.create_dataset('train', (len(train), len(
        train[0])), dtype=train.dtype)[:] = train
    f.create_dataset('test', (len(test), len(
        test[0])), dtype=test.dtype)[:] = test
    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    bf = BruteForceBLAS(distance, precision=train.dtype)

    bf.fit(train)
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print('%d/%d...' % (i, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()


def glove(out_fn, d):
    import zipfile

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X_train, X_test = train_test_split(X)
        write_output(numpy.array(X_train), numpy.array(
            X_test), out_fn, 'angular')


def sift(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
        test = _get_irisa_matrix(t, 'sift/sift_query.fvecs')
        write_output(train, test, out_fn, 'euclidean')


def deep_image(out_fn):
    yadisk_key = 'https://yadi.sk/d/11eDCm7Dsn9GA'
    response = urlopen('https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
        + yadisk_key + '&path=/deep10M.fvecs')
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(',')[0][9:-1]
    filename = os.path.join('data', 'deep-image.fvecs')
    download(dataset_url, filename)



    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = numpy.fromfile(filename, dtype=numpy.float32)
    dim = fv.view(numpy.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]

    X_train, X_test = train_test_split(fv)
    write_output(X_train, X_test, out_fn, 'angular')


def download(src, dst):
    if not os.path.exists(dst):
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


DATASETS = {
    'deep-image-96-angular': deep_image,
    'glove-25-angular': lambda out_fn: glove(out_fn, 25),
    'sift-128-euclidean': sift
}


if __name__ == '__main__':
    for dataset in list(DATASETS.keys()):
    	get_dataset(dataset)
