import io

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import PIL


from plotting import _bootstrap_error


def _gaussian_fit(img):
    assert img.ndim == 2, '_gaussian_fit: Wrong image dimentions'
    assert (img >= 0).all(), '_gaussian_fit: negative image content'
    assert (img > 0).any(), '_gaussian_fit: blank image'
    img_n = img / img.sum()

    mu = np.fromfunction(
        lambda i, j: (img_n[np.newaxis,...] * np.stack([i, j])).sum(axis=(1, 2)),
        shape=img.shape
    )
    cov = np.fromfunction(
        lambda i, j: (
            (img_n[np.newaxis,...] * np.stack([i * i, j * i, i * j, j * j])).sum(axis=(1, 2))
        ) - np.stack([mu[0]**2, mu[0]*mu[1], mu[0]*mu[1], mu[1]**2]),
        shape=img.shape
    ).reshape(2, 2)
    return mu, cov


def _get_val_metric_single(img):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    """
    assert img.ndim == 2, '_get_val_metric_single: Wrong image dimentions'

    img = np.where(img < 0, 0, img)

    mu, cov = _gaussian_fit(img)

    return np.array((*mu, *cov.diagonal(), cov[0, 1], img.sum()))


_METRIC_NAMES = ['Mean0', 'Mean1', 'Sigma0^2', 'Sigma1^2', 'Cov01', 'Sum']


get_val_metric = np.vectorize(_get_val_metric_single, signature='(m,n)->(k)')


def get_val_metric_v(imgs):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    """
    assert imgs.ndim == 3, 'get_val_metric_v: Wrong images dimentions'
    assert (imgs >= 0).all(), 'get_val_metric_v: Negative image content'
    assert (imgs > 0).any(axis=(1, 2)).all(), 'get_val_metric_v: some images are empty'
    imgs_n = imgs / imgs.sum(axis=(1, 2), keepdims=True)
    mu = np.fromfunction(
        lambda i, j: (imgs_n[:,np.newaxis,...] * np.stack([i, j])[np.newaxis,...]).sum(axis=(2, 3)),
        shape=imgs.shape[1:]
    )

    cov = np.fromfunction(
        lambda i, j: (
            (imgs_n[:,np.newaxis,...] * np.stack([i * i, j * j, i * j])[np.newaxis,...]).sum(axis=(2, 3))
        ) - np.stack([mu[:,0]**2, mu[:,1]**2, mu[:,0] * mu[:,1]]).T,
        shape=imgs.shape[1:]
    )

    return np.concatenate([mu, cov, imgs.sum(axis=(1, 2))[:,np.newaxis]], axis=1)


def make_histograms(data_real, data_gen, title, figsize=(8, 8), n_bins=100, logy=False):
    l = min(data_real.min(), data_gen.min())
    r = max(data_real.max(), data_gen.max())
    bins = np.linspace(l, r, n_bins + 1)
    
    fig = plt.figure(figsize=figsize)
    plt.hist(data_real, bins=bins, label='real')
    plt.hist(data_gen , bins=bins, label='generated', alpha=0.7)
    if logy:
        plt.yscale('log')
    plt.legend()
    plt.title(title)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    img = PIL.Image.open(buf)
    return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)


def make_metric_plots(images_real, images_gen, features=None, calc_chi2=False):
    plots = {}
    try:
        metric_real = get_val_metric_v(images_real)
        metric_gen  = get_val_metric_v(images_gen )
    
        plots.update({name : make_histograms(real, gen, name)
                      for name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T)})

        if features is not None:
            if calc_chi2:
                chi2 = 0

            for feature_name, feature in features.items():
                for metric_name, real, gen in zip(_METRIC_NAMES, metric_real.T, metric_gen.T):
                    name = f'{metric_name} vs {feature_name}'
                    if calc_chi2 and (metric_name != "Sum"):
                        plots[name], chi2_i = make_trend(feature, real, gen, name, calc_chi2=True)
                        chi2 += chi2_i
                    else:
                        plots[name] = make_trend(feature, real, gen, name)

    except AssertionError as e:
        print(f"WARNING! Assertion error ({e})")

    if calc_chi2:
        return plots, chi2

    return plots

def calc_trend(x, y, do_plot=True, bins=100, window_size=20, **kwargs):
    assert x.ndim == 1, 'calc_trend: wrong x dim'
    assert y.ndim == 1, 'calc_trend: wrong y dim'

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.7

    if isinstance(bins, int):
        bins = np.linspace(np.min(x), np.max(x), bins + 1)
    sel = (x >= bins[0])
    x, y = x[sel], y[sel]
    cats = (x[:,np.newaxis] < bins[np.newaxis,1:]).argmax(axis=1)
    
    def stats(arr):
        return (
            arr.mean(),
            arr.std() / (len(arr) - 1)**0.5,
            arr.std(),
            _bootstrap_error(arr, np.std)
        )
    
    mean, mean_err, std, std_err, bin_centers = np.array([
        stats(
            y[(cats >= left) & (cats < right)]
        ) + ((bins[left] + bins[right]) / 2,) for left, right in zip(
            range(len(bins) - window_size),
            range(window_size, len(bins))
        )
    ]).T


    if do_plot:
        mean_p_std_err = (mean_err**2 + std_err**2)**0.5
        plt.fill_between(bin_centers, mean - mean_err, mean + mean_err, **kwargs)
        kwargs['alpha'] *= 0.5
        kwargs = {k : v for k, v in kwargs.items() if k != 'label'}
        plt.fill_between(bin_centers, mean - std - mean_p_std_err, mean - std + mean_p_std_err, **kwargs)
        plt.fill_between(bin_centers, mean + std - mean_p_std_err, mean + std + mean_p_std_err, **kwargs)
        kwargs['alpha'] *= 0.25
        plt.fill_between(bin_centers, mean - std + mean_p_std_err, mean + std - mean_p_std_err, **kwargs)

    return (mean, std), (mean_err, std_err)


def make_trend(feature, real, gen, name, calc_chi2=False, figsize=(8, 8)):
    feature = feature.squeeze()
    real = real.squeeze()
    gen = gen.squeeze()

    fig = plt.figure(figsize=figsize)
    calc_trend(feature, real, label='real', color='blue')
    calc_trend(feature, gen, label='generated', color='red')
    plt.legend()
    plt.title(name)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    img = PIL.Image.open(buf)
    img_data = np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)

    if calc_chi2:
        (
            (real_mean, real_std),
            (real_mean_err, real_std_err)
        ) = calc_trend(feature, real, do_plot=False, bins=20, window_size=1)
        (
            (gen_mean, gen_std),
            (gen_mean_err, gen_std_err)
        ) = calc_trend(feature, gen, do_plot=False, bins=20, window_size=1)

        gen_upper = gen_mean + gen_std
        gen_lower = gen_mean - gen_std
        gen_err2 = gen_mean_err**2 + gen_std_err**2

        real_upper = real_mean + real_std
        real_lower = real_mean - real_std
        real_err2 = real_mean_err**2 + real_std_err**2

        chi2 = (
            ((gen_upper - real_upper)**2 / (gen_err2 + real_err2)).sum() +
            ((gen_lower - real_lower)**2 / (gen_err2 + real_err2)).sum()
        )

        return img_data, chi2
    
    return img_data
