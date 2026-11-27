"""Descriptive VaR analysis; minimum-variance weights use the first 500 days."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from fe_model import (load_data, portfolio_weights, min_var_long_only,
    factor_covariance, ewma_variance, ensure_results_dirs, TABLE_DIR, FIG_DIR,
    WINDOW, CONFIDENCES)


def run_static_analysis():
    _, returns_simple = load_data()
    ensure_results_dirs()
    vals=returns_simple.to_numpy(); columns=list(returns_simple.columns)
    fixed=portfolio_weights(returns_simple)
    first_cov=np.cov(vals[:WINDOW],rowvar=False)
    minvar=min_var_long_only(first_cov)
    ports={**fixed,'MinVariance':minvar}
    pd.DataFrame({'asset':columns, 'equal_weight':fixed['EqualWeight'],
        'eq_6040':fixed['6040'], 'min_variance_first500':minvar}).to_csv(
        TABLE_DIR/'portfolio_weights.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    rows=[]; factor_rows=[]
    covariance=np.cov(vals,rowvar=False)
    for name,w in ports.items():
        r=vals@w
        mu=float(r.mean()); std=float(r.std(ddof=1))
        vol_ewma=np.sqrt(ewma_variance(r))
        for a in CONFIDENCES:
            q=np.quantile(r,1-a)
            tail=r[r<=q]
            rows.append({'portfolio':name,'confidence':a,
                'Historical_VaR':max(-q,0.),'Historical_CVaR':max(-tail.mean(),0.),
                'Parametric_VaR':max(norm.ppf(a)*std-mu,0.),
                'RiskMetrics_EWMA_VaR':max(norm.ppf(a)*vol_ewma,0.),
                'portfolio_daily_vol':std})
        for k in (3,7):
            f_cov,_,expl=factor_covariance(vals,fixed_k=k)
            sig=float(np.sqrt(w@f_cov@w));raw_sig=float(np.sqrt(w@covariance@w))
            for a in CONFIDENCES:
                z=norm.ppf(a)
                factor_rows.append({'portfolio':name,'pca_components':k,'pca_cum_explained':expl,
                    'confidence':a, 'port_daily_vol':sig,'raw_cov_vol':raw_sig,
                    'VaR_raw_cov':max(z*raw_sig-mu,0.),
                    'VaR_factor':max(z*sig-mu,0.)})
    summary=pd.DataFrame(rows)
    factors=pd.DataFrame(factor_rows)
    summary.to_csv(TABLE_DIR/'var_summary.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    factors.to_csv(TABLE_DIR/'factor_var_compare.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    av=[]
    for asset in columns:
        r=returns_simple[asset].to_numpy()
        row={'asset':asset,'ann_vol':np.std(r,ddof=1)*np.sqrt(252)}
        for a in CONFIDENCES:
            q=np.quantile(r,1-a)
            row[f'hist_var_{int(a*100)}']=max(-q,0.)
            row[f'hist_cvar_{int(a*100)}']=max(-r[r<=q].mean(),0.)
        av.append(row)
    av=pd.DataFrame(av).sort_values('hist_var_99',ascending=False)
    av.to_csv(TABLE_DIR/'asset_var.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    plot_static(vals@fixed['EqualWeight'],av)
    print('Static descriptive analysis complete (full sample, not OOS).')
    return summary,factors


def plot_static(eq_simple, av):
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,4.8))
    ax.hist(eq_simple*100,bins=65,color='#7294b8',edgecolor='white',alpha=.85)
    for a,c in [(0.95,'#dd9d35'),(0.99,'#b94b45')]:
        cutoff=np.quantile(eq_simple,1-a)*100
        ax.axvline(cutoff,color=c,ls='--',lw=1.5,label=f'{a:.0%} historical VaR: {-cutoff:.2f}%')
    ax.set_title('EqualWeight | Descriptive daily simple returns')
    ax.set_xlabel('Daily simple return (%)');ax.set_ylabel('Days')
    avsort=av.sort_values('hist_var_99')
    ax2.barh(avsort.asset,avsort.hist_var_99*100,color='#7294b8')
    ax2.set_title('ETF 99% historical VaR (full sample)')
    ax2.set_xlabel('1-day VaR (%)')
    fig.subplots_adjust(left=.08, right=.97, top=.71, bottom=.16, wspace=.34)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.94),
               bbox_transform=fig.transFigure, ncol=2, frameon=False, fontsize=9)
    fig.savefig(FIG_DIR/'fe_var_plot.png',dpi=200);plt.close(fig)


if __name__=='__main__':
    run_static_analysis()
