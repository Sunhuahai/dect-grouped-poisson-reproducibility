#!/usr/bin/env python3
"""Verify archived runs, aggregate all seeds, and create manuscript figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dect-matplotlib')
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from dde_2d import METHODS, allocate, group, observed, projected_gradient

ROOT=Path(__file__).resolve().parents[1]
COLORS={'SIRT':'#333333','LS':'#d55e00','WLS':'#009e73','Poisson':'#0072b2'}
LABELS={'SIRT':'SIRT (one step)','LS':'LS solve','WLS':'Weighted LS solve','Poisson':'Proximal Poisson'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'experiments');p.add_argument('--figures',type=Path,default=ROOT/'figures');a=p.parse_args()
    a.figures.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.data/'protocol.json').read_text());c=meta['config'];results=json.loads((a.data/'results.json').read_text());runs=json.loads((a.data/'runs.json').read_text())
    with (a.data/'trajectories.csv').open() as f:
        rows=[{k:(v if k=='method' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
    expected=len(c['fluences'])*c['seeds']*len(METHODS)
    assert len(results)==expected and len(rows)==expected*(c['outer']+1)
    assert len({(r['fluence'],r['seed'],r['method']) for r in results})==expected
    A=sparse.load_npz(a.data/'projector.npz');geo=np.load(a.data/'geometry_truth.npz');truth=geo['Xtrue'];low=np.arange(4)<geo['threshold'][:,None]
    all_curves={};max_identity=0.;max_final_error=0.;all_increases={};summary=[]
    for fluence in c['fluences']:
        for i in range(c['seeds']):
            seed=c['seed_start']+i
            inp=np.load(a.data/f'inputs_f{int(fluence)}_s{seed}.npz');recon=np.load(a.data/f'recon_f{int(fluence)}_s{seed}.npz');y=inp['y'];I0=fluence*np.array([.4,.3,.2,.1])
            rm=next(r for r in runs if r['fluence']==fluence and r['seed']==seed)
            assert hashlib.sha256(y.tobytes()).hexdigest()==rm['observations_sha256']
            for method in METHODS:
                rr=sorted([r for r in rows if r['method']==method and r['fluence']==fluence and r['seed']==seed],key=lambda r:r['iteration'])
                assert [r['iteration'] for r in rr]==list(range(c['outer']+1))
                all_curves[(fluence,seed,method)]=rr
                X=recon[method];assert X.shape==truth.shape and np.all(np.isfinite(X)) and X.min()>=0 and X.max()<=c['upper']
                q=I0*np.exp(-(A@X));z,_=allocate(q,y,low)
                assert np.allclose(group(z,low),y,rtol=1e-13,atol=1e-10)
                f=observed(q,y,low,fluence);err=np.linalg.norm(X-truth)/np.linalg.norm(truth)
                max_final_error=max(max_final_error,abs(f-rr[-1]['objective']),abs(err-rr[-1]['nrmse']))
                assert abs(f-rr[-1]['objective'])<1e-10 and abs(err-rr[-1]['nrmse'])<1e-10
                assert abs(projected_gradient(X,A.T@((z-q)/fluence),c['upper'])-rr[-1]['outer_kkt'])<1e-10
                for prev,r in zip(rr[:-1],rr[1:]):
                    identity=abs(r['surrogate_change']-(r['objective']-prev['objective'])-r['majorizer_gap']);max_identity=max(max_identity,identity)
                    assert identity<1e-8 and r['majorizer_gap']>=-1e-9
                    if method=='Poisson':
                        assert r['sufficient_slack']>=-1e-9
                        assert r['surrogate_change']+.5*c['rho']*r['step']**2<=1e-9
                    if method!='SIRT':assert r['inner_kkt']<=5*c['inner_gtol']
        for method in METHODS:
            rs=[r for r in results if r['fluence']==fluence and r['method']==method]
            s={'fluence':fluence,'method':method}
            for key in ['final_nrmse','final_objective','final_reduced_deviance','final_outer_kkt','seconds','inner_evaluations']:
                vals=[r[key] for r in rs];s[key+'_mean']=float(np.mean(vals));s[key+'_sd']=float(np.std(vals,ddof=1))
            s['objective_increases']=sum(r['objective_increases'] for r in rs)
            summary.append(s)
    (a.data/'summary.json').write_text(json.dumps(summary,indent=2))
    checks=dict(runs=len(runs),method_runs=expected,outer_updates=expected*c['outer'],verified_final_arrays=True,
                max_majorizer_identity_error=max_identity,max_final_metric_error=max_final_error,
                poisson_updates=len(runs)*c['outer'],poisson_minimum_slack=min(r['sufficient_slack'] for r in rows if r['method']=='Poisson' and r['iteration']>0),
                max_inner_kkt=max(r['inner_kkt'] for r in rows if r['method']!='SIRT'),zero_group_total=sum(r['zero_groups'] for r in runs),
                minimum_expected_group=min(r['min_expected_group'] for r in runs),objective_increases={m:sum(r['objective_increases'] for r in results if r['method']==m) for m in METHODS})
    (a.data/'verification.json').write_text(json.dumps(checks,indent=2))
    with (a.data/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    tex=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'$B$ & Update & NRMSE (\%) & $D(X^{100})$ & Time (s) & Increases \\',r'\midrule']
    for s in summary:
        tex.append(f"{int(s['fluence']):,} & { {'SIRT':'SIRT (one step)','LS':'LS solve','WLS':'Weighted LS','Poisson':'Proximal Poisson'}[s['method']]} & "
          +f"${100*s['final_nrmse_mean']:.2f}\\pm{100*s['final_nrmse_sd']:.2f}$ & ${s['final_objective_mean']:.3f}\\pm{s['final_objective_sd']:.3f}$ & "
          +f"${s['seconds_mean']:.3f}\\pm{s['seconds_sd']:.3f}$ & {s['objective_increases']} / {c['seeds']*c['outer']} \\\\")
    tex.extend([r'\bottomrule',r'\end{tabular}']);(a.data/'table_2d.tex').write_text('\n'.join(tex)+'\n')
    def save(fig,name):
        fig.savefig(a.figures/(name+'.pdf'),bbox_inches='tight',pad_inches=.05)
        fig.savefig(a.figures/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.05);plt.close(fig)
    def curve(ax,fluence,metric,xmetric='iteration',scale=1):
        for method in METHODS:
            rr=[all_curves[(fluence,c['seed_start']+i,method)] for i in range(c['seeds'])]
            y=np.array([[r[metric]*scale for r in rs] for rs in rr]);xx=np.mean([[r[xmetric] for r in rs] for rs in rr],axis=0)
            mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
            ax.plot(xx,mean,label=LABELS[method],color=COLORS[method],lw=1.4)
            ax.fill_between(xx,mean-sd,mean+sd,color=COLORS[method],alpha=.13,linewidth=0)
        ax.grid(alpha=.2);ax.set_xlabel('Outer iteration' if xmetric=='iteration' else ('Elapsed time (s)' if xmetric=='seconds' else 'Cumulative inner evaluations'))
    fig,axs=plt.subplots(2,2,figsize=(7.0,4.5),layout='constrained')
    for col,f in enumerate(c['fluences']):
        curve(axs[0,col],f,'objective');axs[0,col].set_yscale('log');axs[0,col].set_ylabel('Normalized half deviance D')
        axs[0,col].set_title(f'B = {int(f):,} incident photons / ray')
        curve(axs[1,col],f,'nrmse',scale=100);axs[1,col].set_ylabel('NRMSE (%)');axs[1,col].legend(loc='best')
    save(fig,'dde_2d_curves')
    fig,axs=plt.subplots(2,2,figsize=(7.0,4.5),layout='constrained')
    for col,f in enumerate(c['fluences']):
        for rr,xm in enumerate(['seconds','inner_evaluations']):
            curve(axs[rr,col],f,'nrmse',xmetric=xm,scale=100)
            axs[rr,col].set_xscale('symlog',linthresh=.1 if xm=='seconds' else 1)
            axs[rr,col].set_ylabel('NRMSE (%)');axs[rr,col].set_title(f'B = {int(f):,}');axs[rr,col].legend()
    save(fig,'dde_2d_time')
    # The first predetermined seed is displayed; it is not selected by performance.
    f=min(c['fluences']);seed=c['seed_start'];rec=np.load(a.data/f'recon_f{int(f)}_s{seed}.npz')
    fig=plt.figure(figsize=(7.,5.1),layout='constrained')
    gs=fig.add_gridspec(4,6,width_ratios=[1,1,1,1,1,.06])
    axs=np.array([[fig.add_subplot(gs[r,k]) for k in range(5)] for r in range(4)])
    for rr,k in enumerate([0,3]):
        axs[2*rr,0].imshow(truth[:,k].reshape(c['n'],c['n']),origin='lower',vmin=0,vmax=3,cmap='gray',interpolation='nearest')
        axs[2*rr,0].set_ylabel(f'Bin {k+1}\nAttenuation');axs[2*rr+1,0].axis('off');axs[2*rr+1,0].text(.5,.5,f'Bin {k+1}\nAbsolute error',ha='center',va='center',transform=axs[2*rr+1,0].transAxes)
        for col,method in enumerate(METHODS,1):
            image=rec[method][:,k].reshape(c['n'],c['n']);tru=truth[:,k].reshape(c['n'],c['n'])
            im=axs[2*rr,col].imshow(image,origin='lower',vmin=0,vmax=3,cmap='gray',interpolation='nearest')
            er=axs[2*rr+1,col].imshow(abs(image-tru),origin='lower',vmin=0,vmax=1.5,cmap='magma',interpolation='nearest')
        for ax in axs[2*rr:2*rr+2,:].ravel():ax.set_xticks([]);ax.set_yticks([])
    for j,title in enumerate(['Truth','SIRT','LS solve','Weighted LS','Prox. Poisson']):axs[0,j].set_title(title)
    fig.colorbar(im,cax=fig.add_subplot(gs[:2,5]),label='Attenuation')
    fig.colorbar(er,cax=fig.add_subplot(gs[2:,5]),label='Absolute error (capped at 1.5)')
    save(fig,'dde_2d_images')
    # All four channels, at both fluences, accompany the main selected-bin figure.
    for f in c['fluences']:
        rec=np.load(a.data/f'recon_f{int(f)}_s{seed}.npz');fig,axs=plt.subplots(4,5,figsize=(7.,5.5),layout='constrained')
        for k in range(4):
            for col,method in enumerate(['Truth']+list(METHODS)):
                xx=truth if method=='Truth' else rec[method]
                im=axs[k,col].imshow(xx[:,k].reshape(c['n'],c['n']),origin='lower',vmin=0,vmax=3,cmap='gray',interpolation='nearest');axs[k,col].set_xticks([]);axs[k,col].set_yticks([])
            axs[k,0].set_ylabel(f'Bin {k+1}')
        for j,title in enumerate(['Truth','SIRT','LS solve','Weighted LS','Prox. Poisson']):axs[0,j].set_title(title)
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label='Attenuation');save(fig,f'dde_2d_allbins_f{int(f)}')
    fig,axs=plt.subplots(1,3,figsize=(7.,2.4),layout='constrained')
    for f,style in zip(c['fluences'],['-','--']):
        rr=[all_curves[(f,c['seed_start']+i,'Poisson')][1:] for i in range(c['seeds'])]
        for j,(metric,label) in enumerate([('step','Step norm'),('outer_kkt','Observed KKT residual'),('sufficient_slack','Sufficient-decrease slack')]):
            yy=np.array([[r[metric] for r in rs] for rs in rr]);xx=np.arange(1,c['outer']+1)
            axs[j].plot(xx,yy.mean(axis=0),style,label=f'B = {int(f):,}',lw=1.4)
            axs[j].fill_between(xx,yy.min(axis=0),yy.max(axis=0),alpha=.12)
            axs[j].set_yscale('log');axs[j].set_xlabel('Outer iteration');axs[j].set_ylabel(label);axs[j].grid(alpha=.2)
    axs[0].legend();save(fig,'dde_2d_diagnostics')
    if a.data.resolve()==(ROOT/'experiments').resolve():
        environment=[ROOT.parent/'.python-version',ROOT.parent/'pyproject.toml',ROOT.parent/'uv.lock']
        sources=sorted((ROOT/'code').glob('*.py'))
        payloads=sorted(p for p in a.data.iterdir() if p.is_file() and p.name!='checksums.sha256')
        lines=[]
        for path in environment+sources+payloads:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f'{digest}  {os.path.relpath(path,ROOT)}')
        (a.data/'checksums.sha256').write_text('\n'.join(lines)+'\n')
    print(json.dumps(checks,indent=2));print('SUMMARY',json.dumps(summary,indent=2))
if __name__=='__main__':main()
