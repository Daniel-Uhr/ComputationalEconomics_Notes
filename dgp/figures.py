import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import binsreg

def plot_test(mu0=0, mu1=3, sigma=1, alpha=0.05, n=100):
    s = np.sqrt(sigma**2 / n)
    x = np.linspace(mu0 - 4*s, mu1 + 4*s, 1000)
    pdf1 = norm(mu0, s).pdf(x)
    pdf2 = norm(mu1, s).pdf(x)
    cv = mu0 + norm.ppf(1 - alpha) * s
    power = norm.cdf(np.abs(mu1 - cv) / s)

    plt.plot(x, pdf1, label=f'Distribution under H0: μ={mu0}');
    plt.plot(x, pdf2, label=f'Distribution under H1: μ={mu1}');
    plt.fill_between(x[x>=cv], pdf1[x>=cv], color='r', alpha=0.4, label=f'Significance: α={alpha:.2f}')
    plt.fill_between(x[x<=cv], pdf2[x<=cv], color='g', alpha=0.4, label=f'β={1-power:.2f}')
    plt.vlines(cv, ymin=0, ymax=plt.ylim()[1], color='k', label=f'Critical Value: {cv:.2f}')
    plt.vlines(mu0, ymin=0, ymax=max(pdf1), color='k', lw=1, ls='--')
    plt.vlines(mu1, ymin=0, ymax=max(pdf2), color='k', lw=1, ls='--')
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.title("Hypothesis Testing")

def make_cmap(color1, color2, K):
    C1 = np.array(mpl.colors.to_rgb(color1))
    C2 = np.array(mpl.colors.to_rgb(color2))
    return [(K-1-k)/(K-1) * C1 + k/(K-1) * C2  for k in range(K)]

def binscatter(data, x, y, by=None, **kwargs):
    est = binsreg.binsreg(data=data, x=x, y=y, **kwargs)
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': x, 'fit': y})
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    if not by is None:
        df_est['group'] = df_est['group'].astype(data[by].dtype)
        df_est = df_est.rename(columns={'group': by})
    return df_est

def binscatterplot(data, x, y, hue=None, **kwargs):
    df_est = binscatter(data=data, x=x, y=y, by=hue, **kwargs)
    plot = sns.scatterplot(x=x, y=y, hue=hue, data=df_est)
    return plot

def xy_from_df(df, r0, r1):
    x = df.iloc[r0:r1,:-1].to_numpy()
    x = np.concatenate((np.ones((np.size(x,0), 1)), x), axis=1)
    y = df.iloc[r0:r1,-1].to_numpy()
    return x, y
