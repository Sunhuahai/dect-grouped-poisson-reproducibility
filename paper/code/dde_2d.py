#!/usr/bin/env python3
"""Reproducible 2-D grouped-Poisson DDE experiment; no imaging data downloads.

Exact pixel intersection projector, four independent image channels, no image
regularizer, common box constraints and initialization. LS/SIRT is the unit
row/column normalized single-step baseline in the project algorithm notes;
LS and WLS solve their frozen log-domain subproblems. Poisson solves the
proximal count-domain subproblem numerically and certifies its decrease.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, time
from dataclasses import dataclass, asdict
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import minimize

METHODS = ('SIRT', 'LS', 'WLS', 'Poisson')
@dataclass
class Config:
    n: int = 32
    views: int = 60
    detectors: int = 48
    outer: int = 100
    init_steps: int = 30
    rho: float = 0.1
    upper: float = 3.0
    inner_gtol: float = 1e-7
    inner_maxiter: int = 1200
    seed_start: int = 20260910
    seeds: int = 5
    fluences: tuple = (20000., 1000.)

def projector(n, views, detectors):
    """Parallel rays over [-1,1]^2; exact piecewise-constant line integrals."""
    edges = np.linspace(-1, 1, n+1)
    offsets = (np.arange(detectors)+.5-detectors/2) * (2*np.sqrt(2)/detectors)
    rr, cc, vv, thresholds, angles, positions = [], [], [], [], [], []
    row = 0
    for a in range(views):
        theta = np.pi*a/views
        normal = np.array([np.cos(theta), np.sin(theta)])
        direction = np.array([-np.sin(theta), np.cos(theta)])
        for offset in offsets:
            origin = offset*normal
            ts=[]
            for k in range(2):
                if abs(direction[k])>1e-12:
                    ts.extend((edges-origin[k])/direction[k])
            ts=np.unique(np.round(ts,14))
            mid=.5*(ts[:-1]+ts[1:]); points=origin[None,:]+mid[:,None]*direction[None,:]
            valid=np.all((points>-1)&(points<1),axis=1)
            lengths=np.diff(ts)[valid]
            if not len(lengths): continue
            ij=np.floor((points[valid]+1)*n/2).astype(int)
            rr.extend([row]*len(lengths));cc.extend((ij[:,1]*n+ij[:,0]).tolist());vv.extend(lengths.tolist())
            thresholds.append(1+a%3);angles.append(a);positions.append(offset);row+=1
    A=sparse.csr_matrix((vv,(rr,cc)),shape=(row,n*n))
    return A,np.asarray(thresholds),np.asarray(angles),np.asarray(positions)

def phantom(n):
    """Two-component spectral variant of the modified Shepp--Logan phantom.

    The first component is the conventional modified Shepp--Logan image.  A
    second, nonnegative contrast component follows its six small positive
    inserts, retaining spatially varying spectra without adding noncanonical
    ellipse locations.  The analytic definition avoids an image dependency.
    """
    coords=(np.arange(n)+.5)*2/n-1
    xx,yy=np.meshgrid(coords,coords)
    # amplitude, horizontal radius, vertical radius, x centre, y centre, angle
    ellipses=np.array([
        [ 1.0, .6900, .9200,  .0000,  .0000,   0.],
        [-.8,  .6624, .8740,  .0000, -.0184,   0.],
        [-.2,  .1100, .3100,  .2200,  .0000, -18.],
        [-.2,  .1600, .4100, -.2200,  .0000,  18.],
        [ .1,  .2100, .2500,  .0000,  .3500,   0.],
        [ .1,  .0460, .0460,  .0000,  .1000,   0.],
        [ .1,  .0460, .0460,  .0000, -.1000,   0.],
        [ .1,  .0460, .0230, -.0800, -.6050,   0.],
        [ .1,  .0230, .0230,  .0000, -.6060,   0.],
        [ .1,  .0230, .0460,  .0600, -.6050,   0.],
    ])
    base=np.zeros((n,n));inserts=np.zeros((n,n))
    for index,(amplitude,a,b,x0,y0,angle) in enumerate(ellipses):
        theta=np.deg2rad(angle);c=np.cos(theta);s=np.sin(theta)
        dx=xx-x0;dy=yy-y0
        mask=((c*dx+s*dy)/a)**2+((-s*dx+c*dy)/b)**2<=1
        base[mask]+=amplitude
        if index>=4:
            inserts[mask]=1.
    m1=np.clip(base,0.,1.)
    m2=.45*inserts
    coeff=np.array([[1.8,1.5,1.25,1.05],[1.5,1.0,.7,.5]])
    X=np.stack((m1,m2),axis=-1).reshape(-1,2)@coeff
    assert X.min()>=0 and X.max()<=3
    return X, np.stack((m1,m2),axis=0), coeff, ellipses

def group(q,low):
    return np.column_stack(((q*low).sum(axis=1),(q*(~low)).sum(axis=1)))

def allocate(q,y,low):
    S=group(q,low)
    z=q*(low*(y[:,0]/S[:,0])[:,None]+(~low)*(y[:,1]/S[:,1])[:,None])
    return z,S

def observed(q,y,low,fluence):
    S=group(q,low)
    # Zero observations contribute S, with 0 log 0 convention.
    v=S-y
    positive=y>0
    v[positive]+=y[positive]*np.log(y[positive]/S[positive])
    return float(v.sum()/fluence)

def projected_gradient(X,g,upper):
    r=g.copy()
    r[(X<=1e-10)&(g>0)]=0
    r[(X>=upper-1e-10)&(g<0)]=0
    return float(np.max(np.abs(r)))

def solve_box(fun,X,cfg):
    """Independent KKT residual checked after L-BFGS-B; failures are visible."""
    shape=X.shape;cur=X.ravel().copy(); evals=0; nit=0; messages=[]
    def wrapped(x):
        nonlocal evals
        evals+=1
        f,g=fun(x.reshape(shape))
        return f,g.ravel()
    for restart in range(3):
        res=minimize(wrapped,cur,jac=True,method='L-BFGS-B',bounds=[(0,cfg.upper)]*cur.size,
                     options=dict(gtol=cfg.inner_gtol,ftol=1e-15,maxiter=cfg.inner_maxiter,maxls=50,maxcor=10))
        cur=res.x;nit+=res.nit;messages.append(str(res.message))
        _,g=fun(cur.reshape(shape)); evals+=1; pg=projected_gradient(cur.reshape(shape),g,cfg.upper)
        if pg<=5*cfg.inner_gtol: break
    if pg>5*cfg.inner_gtol:
        raise RuntimeError(f'Inner KKT residual {pg:g}; {messages}')
    return cur.reshape(shape),pg,evals,nit

def initialize(A,y,I0,cfg):
    target=np.log(I0.sum()/np.maximum(y.sum(axis=1),.5))
    row=np.asarray(A.sum(axis=1)).ravel();col=np.asarray(A.sum(axis=0)).ravel()
    x=np.zeros(A.shape[1]); start=time.perf_counter()
    for _ in range(cfg.init_steps):
        x=np.clip(x-(A.T@((A@x-target)/row))/np.maximum(col,1e-15),0,cfg.upper)
    return np.repeat(x[:,None],4,axis=1), time.perf_counter()-start

def run_one(A,threshold,Xtrue,fluence,seed,cfg,folder):
    low=np.arange(4)[None,:]<threshold[:,None]
    I0=fluence*np.array([.4,.3,.2,.1])
    rng=np.random.default_rng(seed)
    qtrue=I0*np.exp(-(A@Xtrue))
    means=group(qtrue,low)
    y=rng.poisson(means).astype(float)
    X0,init_time=initialize(A,y,I0,cfg)
    runmeta=dict(fluence=fluence,seed=seed,zero_groups=int((y==0).sum()),min_expected_group=float(means.min()),
                 max_expected_group=float(means.max()),initial_nrmse=float(np.linalg.norm(X0-Xtrue)/np.linalg.norm(Xtrue)),
                 init_seconds=init_time,observations_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    np.savez_compressed(folder/f'inputs_f{int(fluence)}_s{seed}.npz',y=y,means=means,X0=X0)
    rows=[]; final={}; reports=[]
    rownorm=np.asarray(A.sum(axis=1)).ravel()[:,None]
    colnorm=np.asarray(A.sum(axis=0)).ravel()[:,None]
    for method in METHODS:
        X=X0.copy(); work=0; start=time.perf_counter()
        q=I0*np.exp(-(A@X));initF=observed(q,y,low,fluence)
        z0,_=allocate(q,y,low)
        initial_kkt=projected_gradient(X,A.T@((z0-q)/fluence),cfg.upper)
        rows.append(dict(method=method,iteration=0,objective=initF,nrmse=runmeta['initial_nrmse'],seconds=0.,inner_evaluations=0,
                         step=0.,outer_kkt=initial_kkt,inner_kkt=0.,majorizer_gap=0.,surrogate_change=0.,sufficient_slack=0.,allocation_change=0.,zero_latents=0,inner_iterations=0))

        for t in range(1,cfg.outer+1):
            old=X.copy(); pold=A@old;qold=I0*np.exp(-pold);z,S=allocate(qold,y,low)
            assert np.allclose(group(z,low),y,rtol=2e-14,atol=1e-11)
            fbefore=observed(qold,y,low,fluence)
            # Log baselines use a documented 0.5-count floor only for exactly zero allocations.
            target=np.log(I0/np.where(z>0,z,.5))
            if method=='SIRT':
                X=np.clip(old-(A.T@((pold-target)/rownorm))/np.maximum(colnorm,1e-15),0,cfg.upper)
                inner_kkt=np.nan; neval=1;nit=1
            elif method in ('LS','WLS'):
                w=np.full_like(z,.25) if method=='LS' else z/fluence
                def fun(U):
                    residual=A@U-target
                    return .5*float(np.sum(w*residual**2)),A.T@(w*residual)
                X,inner_kkt,neval,nit=solve_box(fun,old,cfg)
            else:
                def fun(U):
                    dp=A@(U-old); qnew=qold*np.exp(-dp);d=U-old
                    value=float(np.sum(qold*np.expm1(-dp)+z*dp)/fluence+.5*cfg.rho*np.sum(d*d))
                    return value,A.T@((z-qnew)/fluence)+cfg.rho*d
                X,inner_kkt,neval,nit=solve_box(fun,old,cfg)
            work+=neval
            pnew=A@X;qnew=I0*np.exp(-pnew);znew,_=allocate(qnew,y,low)
            fnew=observed(qnew,y,low,fluence)
            step=float(np.linalg.norm(X-old)); deltaQ=float(np.sum(qold*np.expm1(-(pnew-pold))+z*(pnew-pold))/fluence)
            alphaold=qold/(low*S[:,0,None]+(~low)*S[:,1,None])
            Snew=group(qnew,low);alphanew=qnew/(low*Snew[:,0,None]+(~low)*Snew[:,1,None])
            gap=float(np.sum(z*np.log(alphaold/alphanew))/fluence)
            assert gap>=-1e-9
            assert abs((deltaQ-(fnew-fbefore))-gap)<1e-8*max(1.,abs(deltaQ),abs(fnew))
            slack=fbefore-fnew-.5*cfg.rho*step**2
            if method=='Poisson':
                assert deltaQ+.5*cfg.rho*step**2<=1e-9
                assert slack>=-1e-9
            outer_kkt=projected_gradient(X,A.T@((znew-qnew)/fluence),cfg.upper)
            rows.append(dict(method=method,iteration=t,objective=fnew,nrmse=float(np.linalg.norm(X-Xtrue)/np.linalg.norm(Xtrue)),
                             seconds=time.perf_counter()-start,inner_evaluations=work,step=step,outer_kkt=outer_kkt,inner_kkt=inner_kkt,
                             majorizer_gap=gap,surrogate_change=deltaQ,sufficient_slack=slack,
                             allocation_change=float(np.linalg.norm(znew-z)/max(1.,np.linalg.norm(z))),
                             zero_latents=int((z==0).sum()),inner_iterations=nit))
        final[method]=X
        mr=[r for r in rows if r['method']==method]
        report=dict(method=method,fluence=fluence,seed=seed,final_nrmse=mr[-1]['nrmse'],final_objective=mr[-1]['objective'],
                    final_reduced_deviance=2*fluence*mr[-1]['objective']/y.size,final_outer_kkt=mr[-1]['outer_kkt'],
                    seconds=mr[-1]['seconds'],inner_evaluations=work,objective_increases=int(np.sum(np.diff([r['objective'] for r in mr])>1e-9)),
                    max_inner_kkt=float(max([r['inner_kkt'] for r in mr[1:]])) if method!='SIRT' else None,
                    max_allocation_change=float(max(r['allocation_change'] for r in mr)),
                    minimum_sufficient_slack=float(min(r['sufficient_slack'] for r in mr[1:])) if method=='Poisson' else None)
        reports.append(report)
    for r in rows:r.update(fluence=fluence,seed=seed)
    np.savez_compressed(folder/f'recon_f{int(fluence)}_s{seed}.npz',**final)
    return runmeta,rows,reports

def self_check(A,th,X,cfg):
    rng=np.random.default_rng(1337);u=rng.normal(size=A.shape[1]);v=rng.normal(size=A.shape[0])
    lhs=float(v@(A@u));rhs=float(u@(A.T@v)); adj=abs(lhs-rhs)/max(1,abs(lhs),abs(rhs));assert adj<1e-12
    # Independent analytic chord for horizontal/vertical square intersections.
    B,_,ang,pos=projector(8,4,16);lens=np.asarray(B.sum(axis=1)).ravel()
    test=(ang==0)|(ang==2)
    assert np.max(abs(lens[test]-2))<1e-10
    fluence=1234.;I0=fluence*np.array([.4,.3,.2,.1]);low=np.arange(4)<th[:,None]
    Y=np.clip(.5+.1*rng.normal(size=X.shape),.1,2);q=I0*np.exp(-(A@Y)); y=rng.poisson(group(q,low)).astype(float)
    z,_=allocate(q,y,low);g=A.T@((z-q)/fluence);D=rng.normal(size=X.shape);D/=np.linalg.norm(D);h=1e-5
    fd=(observed(I0*np.exp(-(A@(Y+h*D))),y,low,fluence)-observed(I0*np.exp(-(A@(Y-h*D))),y,low,fluence))/(2*h)
    analytical=float(np.sum(g*D));assert abs(fd-analytical)<1e-7
    return dict(adjoint_relative_error=adj,gradient_directional_error=abs(fd-analytical),square_chord_check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'experiments');p.add_argument('--pilot',action='store_true');args=p.parse_args()
    cfg=Config()
    if args.pilot:cfg.n=24;cfg.outer=25;cfg.seeds=1
    args.out.mkdir(parents=True,exist_ok=True)
    A,th,ang,pos=projector(cfg.n,cfg.views,cfg.detectors);X,materials,coeff,ellipses=phantom(cfg.n)
    checks=self_check(A,th,X,cfg)
    sparse.save_npz(args.out/'projector.npz',A)
    np.savez_compressed(args.out/'geometry_truth.npz',threshold=th,angle_index=ang,offset=pos,Xtrue=X,materials=materials,coeff=coeff,
                        phantom_name='modified Shepp-Logan',shepp_logan_ellipses=ellipses)
    meta=dict(config=asdict(cfg),numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),platform=platform.platform(),
              rays=A.shape[0],voxels=A.shape[1],nnz=A.nnz,checks=checks,
              phantom=dict(name='two-component modified Shepp-Logan',base='standard 10-ellipse modified Shepp-Logan',
                           contrast_component='0.45 times the union of the six positive internal inserts',
                           coefficients=coeff.tolist()),
              notes='Normalized count-domain objective; no spatial or spectral priors. All seeds and predetermined final iteration included.')
    (args.out/'protocol.json').write_text(json.dumps(meta,indent=2))
    runmeta=[];allrows=[];reports=[];log_lines=[]
    for fluence in cfg.fluences:
        for i in range(cfg.seeds):
            m,r,t=run_one(A,th,X,fluence,cfg.seed_start+i,cfg,args.out);runmeta.append(m);allrows+=r;reports+=t
            for report in t:
                line=json.dumps(report);log_lines.append(line);print(line,flush=True)
            for name,obj in [('runs.json',runmeta),('results.json',reports)]:
                (args.out/name).write_text(json.dumps(obj,indent=2))
            with (args.out/'trajectories.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    complete=f'COMPLETE {args.out}';log_lines.append(complete)
    (args.out/'run.log').write_text('\n'.join(log_lines)+'\n')
    print(complete,flush=True)
if __name__=='__main__':main()
