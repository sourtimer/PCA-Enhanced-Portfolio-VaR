"""Figures from saved VaR forecasts. Values are displayed as percentages and legends stay outside the axes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from fe_model import FIG_DIR, MAIN_METHODS

FIGURES = FIG_DIR
MC_FIGURES = FIGURES / 'monte_carlo'

COLORS = {
    'Historical': '#748096',
    'Parametric': '#bf8536',
    'RiskMetrics_EWMA': '#248a7a',
    'PCA_Factor': '#2465a9',
    'MonteCarlo_Normal': '#ab6a96',
    'Actual_Loss': '#263f5b',
    'Violation': '#c4474b',
}
NAMES = {'Historical':'Historical', 'Parametric':'Parametric',
         'RiskMetrics_EWMA':'RiskMetrics EWMA', 'PCA_Factor':'PCA Factor',
         'MonteCarlo_Normal':'Monte Carlo (Normal)'}
METHODS = list(NAMES)


def setup_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':10.3,
        'axes.titlesize':12, 'axes.titleweight':'semibold',
        'axes.labelsize':10, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'axes.edgecolor':'#aebac9', 'axes.labelcolor':'#334155',
        'xtick.color':'#536479', 'ytick.color':'#536479',
        'text.color':'#15243a', 'figure.facecolor':'white', 'axes.facecolor':'white',
        'grid.color':'#d9e2ec','grid.linewidth':.55,'savefig.facecolor':'white',
    })


def style_axis(ax, grid=True):
    for edge in ('top','right'):
        ax.spines[edge].set_visible(False)
    ax.set_axisbelow(True)
    if grid: ax.grid(axis='y', alpha=.68)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def dates(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def legend_top(fig, handles, labels=None, ncol=3, y=0.925, fontsize=9):
    """Top margin is reserved using subplots_adjust; not ax.legend()."""
    leg = fig.legend(handles=handles, labels=labels, loc='upper center',
                     bbox_to_anchor=(.5, y), bbox_transform=fig.transFigure,
                     ncol=ncol, frameon=False, fontsize=fontsize,
                     handlelength=2.35, columnspacing=1.45)
    return leg


def save(fig, target:Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, facecolor='white')
    plt.close(fig)
    return target


def filter_series(daily, portfolio='EqualWeight', confidence=.99):
    out = daily.loc[(daily['portfolio']==portfolio)
                    & np.isclose(daily['confidence'],confidence)].copy()
    if out.empty:
        raise ValueError(f'No observations for {portfolio}, {confidence}')
    out['Date'] = pd.to_datetime(out['Date'])
    return out.sort_values('Date')


def verify_daily(daily):
    keys=['Date','portfolio','confidence','method']
    if daily.duplicated(keys).any():
        raise ValueError('Duplicate model/date/portfolio/confidence predictions')
    required={'VaR','actual_loss','violation','training_end_date'}
    if not required.issubset(daily.columns):
        raise ValueError(f'Missing required columns: {required-set(daily.columns)}')
    if (daily['VaR']<0).any() or not np.isfinite(daily['VaR']).all():
        raise ValueError('Invalid VaR in validated predictions')
    breaches=(daily['actual_loss']>daily['VaR']).astype(int)
    if not np.array_equal(breaches.to_numpy(), daily['violation'].astype(int).to_numpy()):
        raise ValueError('Violation flags inconsistent with underlying observed losses')
    if not (pd.to_datetime(daily['training_end_date']) < pd.to_datetime(daily['Date'])).all():
        raise ValueError('Training end date leaks into forecast date')


def plot_rolling_var(daily, outfile=None):
    """Revised original rolling_var_plot; all legends outside axes."""
    setup_style()
    sub = filter_series(daily, 'EqualWeight', .99)
    pivot = sub.pivot(index='Date', columns='method', values='VaR')
    reference = sub.loc[sub['method']=='Historical'].set_index('Date').sort_index()
    fig, (ax,ax2)=plt.subplots(2,1,sharex=True,figsize=(14.5,8.4),
                                gridspec_kw={'height_ratios':[1.55,1.0]})
    fig.subplots_adjust(left=.09,right=.96,bottom=.11,top=.79,hspace=.24)
    fig.suptitle('Rolling one-day 99% VaR | Equal-weight ETF portfolio',
                 fontsize=18,fontweight='bold',x=.09,ha='left',y=.965)
    fig.text(.09,.922,'1,194 out-of-sample forecasts · Percent of portfolio value · 2021–2026',
             fontsize=10,color='#62758a')
    handles=[]
    for m in MAIN_METHODS:
        line,=ax.plot(pivot.index,pivot[m]*100,color=COLORS[m],lw=1.4,
                      label=NAMES[m])
        handles.append(line)
    ax.set_ylabel('Predicted VaR (%)')
    ax.set_title('A   Estimated risk thresholds', loc='left',pad=10)
    style_axis(ax)
    actual_loss=reference.actual_loss*100
    line,=ax2.plot(reference.index, actual_loss, color=COLORS['Actual_Loss'],
                   lw=.86,alpha=.82,label='Observed daily loss')
    br=reference.loc[reference.violation==1]
    dots=ax2.scatter(br.index,br.actual_loss*100,marker='o',s=25,alpha=.98,
                     color=COLORS['Violation'],edgecolor='white',linewidth=.45,zorder=5,
                     label=f'Historical breaches (n={len(br)})')
    ax2.axhline(0,color='#93a5b8',lw=.8)
    ax2.set_title('B   Observed gains/losses and Historical VaR exceedances',loc='left',pad=10)
    ax2.set_ylabel('Observed loss (%)')
    ax2.set_xlabel('Forecast date')
    style_axis(ax2); dates(ax2)
    legend_top(fig,handles+[line,dots],ncol=3,y=.888)
    fig.text(.09,.038,'Note: Positive observed loss = portfolio decline; negative values = gain. Breaches refer to the Historical model.',
             fontsize=8.5,color='#62758a')
    return save(fig,Path(outfile) if outfile else FIGURES/'rolling_var_plot.png')


def plot_monte_comparison(daily,outfile=None):
    """Five-method VaR series + dedicated MC-vs-parametric difference panel."""
    setup_style()
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(15,10.8),
                          gridspec_kw={'height_ratios':[1.06,1.06,.73]})
    fig.subplots_adjust(left=.09,right=.965,bottom=.105,top=.795,hspace=.28)
    fig.suptitle('Five-model risk forecasts | Equal-weight portfolio',
                 fontsize=18,fontweight='bold',x=.09,ha='left',y=.965)
    fig.text(.09,.924,'One-day 95% and 99% VaR · 500-day rolling training window · Same 1,194 forecast dates',
             fontsize=10,color='#62758a')
    legend_handles=[]
    for i,c in enumerate([.95,.99]):
        g=filter_series(daily,'EqualWeight',c)
        pivot=g.pivot(index='Date',columns='method',values='VaR').sort_index()
        ax=axes[i]
        for method in METHODS:
            line,=ax.plot(pivot.index,pivot[method]*100,label=NAMES[method],
                           color=COLORS[method],lw=1.25 if method!='MonteCarlo_Normal' else 1.6,
                           ls='--' if method=='MonteCarlo_Normal' else '-',
                           alpha=.95)
            if i==0: legend_handles.append(line)
        ax.set_ylabel('VaR (% loss)')
        ax.set_title(f'{"A" if i==0 else "B"}   {c:.0%} VaR forecasts',loc='left',pad=10)
        style_axis(ax)
        if i==1: diff99=(pivot['MonteCarlo_Normal']-pivot['Parametric'])*1e4
    ax=axes[2]
    ax.plot(diff99.index,diff99.to_numpy(),color='#745da2',lw=1.15)
    ax.axhline(0,color='#8c9bab',lw=1,linestyle='--')
    ax.set_ylabel('Difference (bp)')
    ax.set_title('C   Monte Carlo minus Parametric VaR (99% confidence)',loc='left',pad=10)
    style_axis(ax); dates(ax); ax.set_xlabel('Forecast date')
    legend_top(fig,legend_handles,ncol=5,y=.879,fontsize=9)
    fig.text(.09,.038,'Note: Monte Carlo and Parametric both assume normal returns; their differences are primarily simulation noise. 1 bp = 0.01 percentage point.',
             fontsize=8.5,color='#62758a')
    return save(fig,Path(outfile) if outfile else MC_FIGURES/'monte_carlo_compare.png')


def plot_breach_rates(summary,outfile=None):
    """Present original 2x3 comparison with a SINGLE figure-level legend."""
    setup_style()
    portfolios=['EqualWeight','6040','MinVariance']
    portlabels={'EqualWeight':'Equal weight','6040':'60/40','MinVariance':'Minimum variance'}
    fig,axes=plt.subplots(2,3,sharey='row',figsize=(15.5,9.2))
    fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.16,hspace=.42,wspace=.19)
    fig.suptitle('Backtest exceedance rates across five VaR models',
                 fontsize=18,fontweight='bold',x=.07,ha='left',y=.965)
    fig.text(.07,.923,'Bars: observed violations  ·  Dashed line: theoretical target  ·  1,194 days per combination',
             fontsize=10,color='#62758a')
    names=['Historical','Parametric','EWMA','PCA','MC Normal']
    for ri,c in enumerate([.95,.99]):
        vals=summary.loc[np.isclose(summary['confidence'],c)]
        upper=max(float(vals.actual_breach_rate.max())*100,(1-c)*100)*1.23
        for ci,portfolio in enumerate(portfolios):
            ax=axes[ri,ci]
            ss=vals.loc[vals.portfolio==portfolio].set_index('method').reindex(METHODS)
            numbers=ss.actual_breach_rate.to_numpy()*100
            ax.bar(np.arange(5),numbers,color=[COLORS[m] for m in METHODS],width=.7)
            ax.axhline((1-c)*100,color='#b44d51',lw=1.5,ls='--')
            ax.set_ylim(0,upper)
            ax.set_xticks(range(5), names,rotation=0)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_title(f'{"A" if ri==0 else "B"}{ci+1}   {portlabels[portfolio]} · {c:.0%}',loc='left',pad=9)
            if ci==0: ax.set_ylabel('Actual breach rate (%)')
            style_axis(ax)
            for x,v in enumerate(numbers):
                ax.text(x,v+upper*.021,f'{v:.2f}',ha='center',fontsize=8,color='#334155')
    handles=[Line2D([0],[0],lw=6,color=COLORS[m]) for m in METHODS]
    handles.append(Line2D([0],[0],color='#b44d51',lw=1.5,ls='--'))
    legend_top(fig,handles,[NAMES[m] for m in METHODS]+['Expected rate'],ncol=3,y=.882,fontsize=9)
    fig.text(.07,.045,'Target rate: 5% for 95% VaR; 1% for 99% VaR. Similar breach rates do not imply identical model performance.',
             fontsize=8.5,color='#62758a')
    return save(fig,Path(outfile) if outfile else MC_FIGURES/'breach_rate_comparison.png')


def plot_actual_loss_vs_var(daily,outfile=None):
    """Key new presentation figure with red breach dates; zoom chosen by largest observed loss."""
    setup_style()
    g=filter_series(daily,'EqualWeight',.99)
    pivot=g.pivot(index='Date',columns='method',values='VaR').sort_index()
    ref=g.loc[g.method=='PCA_Factor'].set_index('Date').sort_index()
    observed=ref.actual_loss*100
    threshold=pivot['PCA_Factor']*100
    breaches=observed>threshold
    if int(breaches.sum())!=int(ref.violation.sum()):
        raise ValueError('PCA breaches have inconsistent counts')
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(15.6,9.0),
                             gridspec_kw={'height_ratios':[1.14,1]},sharey=False)
    fig.subplots_adjust(left=.085,right=.97,bottom=.105,top=.785,hspace=.31)
    fig.suptitle('When did the risk model fail?',fontsize=18,fontweight='bold',x=.085,ha='left',y=.962)
    fig.text(.085,.921,f'Equal-weight portfolio · PCA Factor · 99% one-day VaR · {int(breaches.sum())} violations in {len(ref):,} forecast days',
             fontsize=10,color='#62758a')
    def draw(ax, date_slice, label):
        o=observed.loc[date_slice]; t=threshold.loc[date_slice]
        bh=(o>t)
        line1,=ax.plot(o.index,o.values,color=COLORS['Actual_Loss'],lw=.9,alpha=.8,label='Observed loss')
        line2,=ax.plot(t.index,t.values,color=COLORS['PCA_Factor'],lw=1.75,label='Predicted 99% VaR')
        points=ax.scatter(o.index[bh],o[bh].values,color=COLORS['Violation'],s=38,
                          edgecolor='white',lw=.5,zorder=6,label='VaR violation')
        ax.axhline(0,color='#93a5b8',lw=.9)
        ax.set_title(label,loc='left',pad=8)
        ax.set_ylabel('Portfolio loss (%)')
        style_axis(ax); dates(ax)
        return [line1,line2,points]
    handles=draw(ax,slice(None),'A   Full out-of-sample period')
    biggest=observed.idxmax()
    center_loc=observed.index.get_loc(biggest)
    left=max(0,center_loc-65); right=min(len(observed),center_loc+66)
    focus_index=observed.index[left:right]
    draw(ax2,slice(focus_index[0],focus_index[-1]),f'B   131-day focus around highest observed loss ({biggest:%d %b %Y})')
    ax2.set_xlabel('Forecast date')
    legend_top(fig,handles,ncol=3,y=.873)
    fig.text(.085,.042,'Loss is positive when portfolio value falls. Red markers denote actual loss > same-day PCA VaR; focus interval selected mechanically.',
             fontsize=8.5,color='#62758a')
    return save(fig,Path(outfile) if outfile else MC_FIGURES/'actual_loss_vs_var.png')


def plot_pca_vs_parametric(daily,outfile=None):
    """Key new presentation figure; side-by-side VaR forecast plus signed bps gap."""
    setup_style()
    g=filter_series(daily,'EqualWeight',.99)
    pivot=g.pivot(index='Date',columns='method',values='VaR').sort_index()
    base=pivot['Parametric']*100
    pca=pivot['PCA_Factor']*100
    delta_bps=(pivot['PCA_Factor']-pivot['Parametric'])*1e4
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(14.5,8.4),sharex=True,
                             gridspec_kw={'height_ratios':[1.45,.85]})
    fig.subplots_adjust(left=.09,right=.965,bottom=.12,top=.78,hspace=.26)
    fig.suptitle('Does PCA change the VaR forecast?',fontsize=18,fontweight='bold',x=.09,ha='left',y=.966)
    fig.text(.09,.925,'Equal-weight portfolio · 99% one-day VaR · Rolling PCA (90% explained-variance threshold)',
             fontsize=10,color='#62758a')
    l1,=ax.plot(pivot.index,base,color=COLORS['Parametric'],lw=1.65,label='Full-covariance Parametric')
    l2,=ax.plot(pivot.index,pca,color=COLORS['PCA_Factor'],lw=1.3,ls='--',label='PCA Factor')
    ax.set_ylabel('Predicted VaR (%)')
    ax.set_title('A   Model forecasts over the same 1,194 days',loc='left',pad=10)
    style_axis(ax)
    ax2.plot(pivot.index,delta_bps,color='#745da2',lw=1.1)
    ax2.axhline(0,color='#91a1b4',lw=1,ls='--')
    ax2.fill_between(pivot.index,delta_bps,0,color='#745da2',alpha=.1)
    ax2.set_ylabel('PCA − Parametric (bp)')
    ax2.set_xlabel('Forecast date')
    ax2.set_title('B   Signed daily difference reveals what overlapping curves conceal',loc='left',pad=10)
    style_axis(ax2);dates(ax2)
    legend_top(fig,[l1,l2],ncol=2,y=.873)
    mad=float(np.mean(np.abs(delta_bps)))
    maximum=float(np.max(np.abs(delta_bps)))
    fig.text(.09,.041,f'Mean absolute gap: {mad:.2f} bp   |   Maximum absolute gap: {maximum:.2f} bp.  Positive difference means higher PCA VaR.',
             fontsize=9,color='#52657c')
    return save(fig,Path(outfile) if outfile else MC_FIGURES/'pca_vs_parametric.png')


def build_all(daily,summary, include_rolling=True):
    """Create ALL updated / added figures without refitting risk models."""
    verify_daily(daily)
    outputs=[]
    if include_rolling: outputs.append(plot_rolling_var(daily))
    outputs += [plot_monte_comparison(daily),plot_breach_rates(summary),
                plot_actual_loss_vs_var(daily),plot_pca_vs_parametric(daily)]
    return outputs
