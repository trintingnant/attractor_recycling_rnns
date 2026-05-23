import numpy
from matplotlib import pyplot as plt, patches
from matplotlib.gridspec import GridSpec
import torch
from typing import Optional, Union, List
import rnn
from rnn import RNN, run_rnn
from default_params import default_params
import numpy as np
from sklearn.decomposition import PCA


#PATHS
import sys
sys.path.append('../FixPointSearch')
from FixPointSearch.FixedPoint import FixedPoint 
#add style sheets to path
sys.path.append('../styles')
import plotstyle

NUM_ANGLES = 25

#=======GENERAL UTILITY FUNCTIONS FOR AX MANIPULATION========

def center(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.scatter(0,0, c='black', alpha=.5)

    ax.set(xticks=[], yticks=[])

def make_naked(ax):
    ax.axis('off')

#=======REGULAR PLOT UTILS=====================

def plot_spectrum(model, w_rec=True, mat: Optional[np.ndarray]=None, **kwargs):

    """Plots the eigenvalue spectrum of a (RNN-weight)-matrix"""

    from matplotlib.patches import Circle, Arc
    
    if w_rec:
        w_rec = model.rnn.w_rec.detach().numpy()
    else:
        assert mat is not None
        w_rec = mat
    
    eigvecs, eigvals = np.linalg.eig(w_rec)[::-1]
    real, imag = np.real(eigvals), np.imag(eigvals)
    fig, ax = plt.subplots()
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
        left=False)
    ax.set_ylabel(r"$Im(\lambda$)")
    ax.set_xlabel(r"$Re(\lambda$)")
    ax.set_aspect("equal", adjustable="box")
    r = 2*model.params["spectral_radius"]
    
    """
    arc = Arc((0,0), r, r, 
              color='silver', 
              theta1=330, 
              theta2=30, 
              linewidth=5,
              linestyle='--',
              label=r"$g$")
    arc2 = Arc((0,0), 2, 2, 
              color='black', 
              theta1=330, 
              theta2=30, 
              linewidth=5,
              linestyle='--',
              label="stability line")
    
    
    ax.add_patch(arc)
    ax.add_patch(arc2)
    """

    c = [1 if r > 1 else 0 for r in real]
    circle = Circle((0,0), model.params["spectral_radius"], fill=True, alpha=0.2, color="black", 
                    label=r"$g={0}$".format(model.params["spectral_radius"]), zorder=-1)
    ax.add_artist(circle)
    ax.scatter(real, imag, c=c, alpha=.8, zorder=10)
    #plot stability li
    ymin, ymax = 1.2*np.min(imag), 1.2*np.max(imag)
    xmin, xmax = 1.2*np.min(real), 1.2*np.max(real)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.axvline(x=1, ymin=ymin, ymax=ymax,  c="black", linestyle='--', label="stability line", alpha=.5)
    ax.axvline(x=0, ymin=ymin, ymax=ymax, c="grey", alpha=.2)
    ax.axhline(y=0, c="grey", alpha=.2)
    ax.scatter(0,0, facecolors=None, c='grey', alpha=.2, zorder=-1)
    ax.plot()
    ax.legend()
    

    plt.show()
    return fig, ax

def plot_multitask_net_response(model, trial, antitrial):
    fig, axs = plt.subplots(2,1, sharey=False, sharex=True) 
    trial, target, _ = trial
    antitrial, antitarget, _ = antitrial

    rates, out = model(trial.unsqueeze(dim=0))
    antirates, antiout = model(antitrial.unsqueeze(dim=0))
    axs[0].set_xlim(0, 200)
    axs[0].plot(out.detach().numpy()[0,:,:], c="b", alpha=.5)
    axs[0].plot(target, "-.", c="black", alpha=.5)
    axs[1].set_title('Antitask')
    axs[1].plot(antiout.detach().numpy()[0,:,:], c="r", alpha=.5)
    axs[1].plot(antitarget, "-.", c="black", alpha=.5)

    axs[0].set_title('Task')
    plt.tight_layout(w_pad=1.5, h_pad=1.5)
    plt.show() 

    return axs

def plot_common_projections(model, 
                            trial, 
                            antitrial, 
                            dim_p=np.array([0,1]), 
                            trnst=80, 
                            n_components=6, 
                            t=20, 
                            dt=1, 
                            shutoff=1, 
                            asp_ratio=10, 
                            ax=None
    ):

    rates, _ = model(trial)
    antirates, _ = model(antitrial)
    combined = np.concatenate((rates.detach().numpy()[:, trnst:-shutoff, :], antirates.detach().numpy()[:,trnst:-shutoff,:]))
    pcs, proj, cumvar = get_PCA(combined, n_components=n_components)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
        left=False)
    
    projs = proj[..., dim_p] 
    projs = projs.reshape(2,-1,2)
    ax.plot(*projs[1,:].T, alpha=.6, label='task')
    ax.plot(*projs[0,:].T, alpha=.6, label='antitask')
    

    #add arrows to show direction of rotation
    axWidth = ax.get_xlim()[1] - ax.get_xlim()[0]
    axHeigth = ax.get_ylim()[1] - ax.get_ylim()[0] 
    ax.arrow(*projs[1,t], *(projs[1,t+dt]-projs[1,t]), head_width=axWidth/asp_ratio, lw=0, alpha=.8)
    ax.arrow(*projs[0,t], *(projs[0,t+dt]-projs[0,t]), head_width=axWidth/asp_ratio, lw=0, alpha=.8)
    
    return ax



def create_common_projections_grid(model, 
                                    trial, 
                                    antitrial, 
                                    asp_ratio=15
    ):

    fig = plt.figure(layout='tight')
    grid = GridSpec(4,4, figure=fig)
    for i in range(4):
        for j in range(i+1, 4): 
            axarr = fig.add_subplot(grid[i,j])
            axarr.yaxis.set_label_position('right')
            if i==0:
                axarr.xaxis.set_label_position('top')
                axarr.set_xlabel(f'PC{j+1}')
            if j==3:
                axarr.set_ylabel(f'PC{i+1}')
            plot_common_projections(model, 
                                    trial, 
                                    antitrial, 
                                    dim_p=np.array([i,j]), 
                                    asp_ratio=15, 
                                    ax=axarr)

    fig.show()

def plot_input_interpolation_fr(model,
                                inputs: Optional,
                                fps: Optional=None,
                                dims=[0,1,2],
                                trnst=50,
                                **kwargs
                                ):

    from utils import gram_schmidt_vecs, get_unstable_subspace

    use_pca_3 = kwargs.get('use_pca_3', False)
    use_unstable_subspace = kwargs.get('unstable_subspace', False)

    if use_unstable_subspace:
        unstable_subspace = get_unstable_subspace(model, orth=True)

    assert not(use_pca_3 and use_unstable_subspace)
    i, j, k = dims
    n_components=kwargs.get('n_components')
    if not n_components:
        n_components = 3
    
    rates, out = model(inputs[0, :, :].unsqueeze(dim=0))
    ratess = torch.zeros((inputs.shape[0], *rates.shape[1:]))
    for inputs_idx in range(inputs.shape[0]):
        rates, out = model(inputs[inputs_idx, :, :].unsqueeze(dim=0))
        ratess[inputs_idx, :, :] = rates.squeeze()
    pca = PCA(n_components=n_components)
    ratess_new = ratess[:,trnst:,:].detach().numpy().reshape(inputs.shape[0]*(rates.shape[1]-trnst), -1)
    pca.fit(ratess_new)

    if not use_pca_3:
        #get new basis:
        if use_unstable_subspace:
            basis = get_unstable_subspace(model)
            orth_basis = gram_schmidt_vecs(basis)
        else:
            sine_out = model.rnn.w_out[:,0].detach().numpy()
            basis = np.vstack((*pca.components_[:2,:], sine_out))
            orth_basis = gram_schmidt_vecs(basis)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    ax.grid(False)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
        left=False)

    ax.set_xlabel("PC 1" if not use_unstable_subspace else "Vec 1", labelpad=-10)
    ax.set_ylabel("PC 2" if not use_unstable_subspace else "Vec 2", labelpad=-12)

    if use_pca_3:
        ax.set_zlabel("PC 3", labelpad=-15, rotation=90)

    elif use_unstable_subspace:
            ax.set_zlabel("Vec 3", labelpad=-15, rotation=90)
    else:
        ax.set_zlabel(r"$\mathbf{w}_{o, \perp}$", labelpad=-15, rotation=90, size=15)

    for inputs_idx in range(inputs.shape[0]):
        
        rates = ratess[inputs_idx,:,:].detach().numpy()
        proj_fp_rates = pca.transform(rates) if use_pca_3 else ((unstable_subspace @ rates.T).T if use_unstable_subspace
                            else (orth_basis @ rates.T).T)
        
        ax.plot(proj_fp_rates[trnst:,i],
                proj_fp_rates[trnst:,j],
                proj_fp_rates[trnst:,k],
                alpha=1,
                zorder= np.abs(np.floor(NUM_ANGLES/2) - inputs_idx)
               )
    if fps:
        for idx, fp in enumerate(fps):
            fp_x_star = fp.x_star.detach().numpy()
            Jdx = fp.Jdx_star.clone().detach().numpy()
            eigvals, eigvecs = np.linalg.eig(Jdx)
            
            #only consider complex conjugates -> for intermediate cases real eigenvalue actually becomes larger
            #-> equilibrium is no longer de-stabilised
            eigvecs = eigvecs[np.iscomplex(eigvals)][:,:2]
            proj_fp_x_star = pca.transform(fp_x_star).T if use_pca_3 else (orth_basis @ fp_x_star.T)
            ax.scatter(proj_fp_x_star[i,0],
                       proj_fp_x_star[j,0],
                       proj_fp_x_star[k,0],
                       facecolors='none',
                       edgecolors='black',
                       marker='o',
                       alpha=.5,
                       zorder=NUM_ANGLES+1
                      )
            
            ax.scatter(proj_fp_x_star[i,0],
                       proj_fp_x_star[j,0],
                       proj_fp_x_star[k,0],
                       facecolors='black' if np.real(eigvals[0])<1 else 'none',
                       edgecolors='black',
                       marker='o',
                       alpha=.5,
                       zorder=NUM_ANGLES+1
                      )
        idxs = kwargs.get('eigvec_idxs')
        if not idxs:
            idxs = [0,1,2,10,11,12,-3,-2,-1]

        #only plot these:
        if idx in idxs:
            Jdx = fp.Jdx_star.clone().detach().numpy()
            eigvals, eigvecs = np.linalg.eig(Jdx)
            #only consider complex conjugates -> for intermediate cases real eigenvalue actually becomes larger
            #-> equilibrium is no longer de-stabilised
            eigvecs = eigvecs[:, np.iscomplex(eigvals)][:,:2]
            
            real = np.real(eigvecs)[:,0]
            imag = np.imag(eigvecs)[:,0]
            real = pca.transform(real).T if use_pca_3 else (orth_basis @ real.T)
            imag = pca.transform(imag).T if use_pca_3 else (orth_basis @ imag.T)

            quiveropts = dict(color='white', headlength=0, pivot='middle', scale=3, 
            linewidth=.5, units='xy', width=.05, headwidth=1)
        
        
            ax.quiver(proj_fp_x_star[i,0],
                  proj_fp_x_star[j,0],
                  proj_fp_x_star[k,0],
                  real[i]*3.5,
                  real[j]*3.5,
                  real[k]*3.5,
                  color="black",
                  linewidth=1)

            ax.quiver(proj_fp_x_star[i,0],
                  proj_fp_x_star[j,0],
                  proj_fp_x_star[k,0],
                  imag[i]*3.5,
                  imag[j]*3.5,
                  imag[k]*3.5,
                  color="black",
                  linewidth=1)

def plot_vector_field_pca(input_=None, 
                            trnst=50,
                            xlims=[-6,6],
                            ylims=[-6,6],
                            dims=[0,1],
                            model=None,
                            num_points=50,
                            fp: FixedPoint=None,
                            pca: PCA=None,
                            **kwargs
                            ):

    """
    Plots vector fields in the space spanned by the first two PCs of the input-induced dynamics.

    INFO: a problem arises when comparing the plots for different input conditions as computing the PCs for each 
    input individual does not necessarily reveal a counter-rotation. This is because the task is solved by 'rotating
    the unstable subspace for each input, which results in equally rotated PCs. Trajectories can therefore look as though
    they are co-rotating when in fact they are not. For this reason, a common PCA object can be passed. This will usually
    yield the desired counter-rotation, but also cause the flow fields to look a lot messier.

    In any case, the relevant pca object should be created before execution and passed as an argument.
    """
    fig, axs = plt.subplots(1,1)
    
    w_rec = model.rnn.w_rec.clone().detach().numpy()
    w_in = model.rnn.w_in.clone().detach().numpy() 

    if fp is not None:
        x_star = fp.x_star.detach().numpy()

    rates, _ = model(input_)
    rates = rates[:,trnst:,:].detach().numpy()
    rates = rates.reshape(-1, rates.shape[2])


    
    if pca is None:
        num_components = kwargs.get("num_components")
        if num_components is None:
            #set to something safe and ridiculous
            num_components = 20
        #perform regular (independent for each input) PCA.
        pca = PCA(n_components=num_components)
        pca.fit(rates)
    
    comp1, comp2 = pca.components_[:2,:]


    #obtain singular values for scaling the grid
    sc1, sc2 = np.sqrt(pca.singular_values_[dims])

    #input_ = torch.from_numpy(np.expand_dims(fp.input_, 1))\
    #if fp is not None else input_

    I = input_.matmul(model.rnn.w_in).detach().numpy()[0,0,:]
    #print(f"input_shape: {I.shape}")
    #TODO: use gram_schmidt directly here
    I_orth = I - (I @ comp1) * comp1 / (comp1 @ comp1) - \
            (I @ comp2) * comp2 / (comp2 @ comp2)

    #create grid on which to evaluate flow field
    xs = np.linspace(*xlims, num_points)
    ys = np.linspace(*ylims, num_points)

    #set up flow field
    vfield = np.zeros((num_points, num_points, 2))

    #dynamix
    def dxdt(x, I=I):
        dxdt = -x + w_rec @ np.tanh(x) + I
        return dxdt

    #define inverse for transformation grid (Euclidean) -> PCA space
    def inverse_transform(mesh_point, components, rates):
        proj = mesh_point @ components
        use_unstable_subspace = kwargs.get("unstable_subspace")
        return proj if use_unstable_subspace else proj + rates.mean(axis=0)

    #curry func
    inverse_transform_curr = lambda z: inverse_transform(z,
                                                np.array([comp1, comp2]),
                                                rates)
    """
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
        left=False)
    """

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = np.array([x,y])
            transformed_point = inverse_transform_curr(z)
            flow = dxdt(transformed_point)
            vfield[j,i, 0] = (flow @ comp1) 
            vfield[j,i, 1] = (flow @ comp2) 
    axs.streamplot(ys, xs, vfield[:,:,0], vfield[:,:,1], color='white', zorder=0)
    norm = np.sqrt(vfield[:,:,0]**2 + vfield[:,:,1]**2) 
    X, Y = np.meshgrid(xs, ys)
    axs.pcolor(X, Y, norm, cmap='bone', zorder=-1)

    rates = pca.transform(rates)
    axs.plot(rates[:,0], rates[:,1],
    c='coral', linewidth=1)

    print(rates[-1,0], rates[-1,1])

    if fp is not None:
        fp_xstar = fp.x_star.detach().numpy().squeeze()
        axs.scatter((fp_xstar - rates.mean(axis=0)) @ comp1, \
        (fp_xstar - rates.mean(axis=0)) @ comp2, \
        c="coral", marker="x", zorder=1)

    try:
        plot_rates = kwargs["plot_rates"]
        rates = pca.transform(rates)
        axs.plot(rates[:,0], rates[:,1],
        c='coral', linewidth=1)
        

    except:
        pass

    return axs

def plot_vector_field_MF(model, 
                         input, 
                         trnst=0,
                         num_points=50,
                         xlims=[-1,1],
                         ylims=[-1,1],
                         find_fps=False,
                         num_fp_trials=20,
                         dist_thresh=1e-2,
                         ax=None
                         ):
    
    from utils import compute_deltas, g
    plotstyle.use("general", 25)
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    ax.set_xlabel(r"$\kappa_{1}$")
    ax.set_ylabel(r"$\kappa_{2}$")
    """
    ax.tick_params(left=False, 
                   bottom=False, 
                   labelleft=False, 
                   labelbottom=False)
    """

    assert (model.rnn.rank == 2)
    r = model.rnn.rank
    in_dim = model.rnn.in_dim
    weights = model.rnn.weights.detach().numpy()
    covs = model.rnn.covs.detach().numpy()
    nidx = r
    Iidx = 2*r
    Widx = 2*r + in_dim

    #create grid on which to evaluate flow field
    xs = np.linspace(*xlims, num_points)
    ys = np.linspace(*ylims, num_points)

    #set up flow field
    vfield = np.zeros((num_points, num_points, 2))
    #dynamix
    def dkdt(k) :
    
        if len(k.shape)==1:
            k = k.reshape(1, k.shape[0])
        K = k[:,:2]
        V = k[:,2:]
        rec_input = np.stack(
    						(np.einsum('...ijk, ...lj -> ...lik', covs[nidx:Iidx, :nidx,:], K),
     						np.einsum('...ijk, ...lj -> ...lik', covs[nidx:Iidx, Iidx:Widx,:], V) 
     						)
						)
        
        gains = g(compute_deltas(model, k))
        rec_input = (rec_input * weights * gains).sum(axis=(0,3))
        dkdt = -K + rec_input
        return dkdt
   
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = np.array([x,y])
            k = np.hstack((z, input[0,:]))[np.newaxis,:]
            flow = dkdt(k)
            vfield[j,i, 0] = flow[:,0][0]
            vfield[j,i, 1] = flow[:,1][0]

    ax.streamplot(ys, xs, vfield[:,:,0], vfield[:,:,1], color='white', zorder=0)
    norm = np.sqrt(vfield[:,:,0]**2 + vfield[:,:,1]**2)
    X, Y = np.meshgrid(xs, ys)
    ax.pcolor(X, Y, norm, cmap='bone', zorder=-1)
    rates, _ = run_rnn(model, torch.from_numpy(input))
    ax.plot(rates[0,trnst:,0], rates[0,trnst:,1], c='hotpink')

    #find fixed points, i.e. roots of dkdt
    if find_fps:
            
        def DKDT(k, input):
            """
            This is the function passed to the root solver
            """

            k_full = torch.from_numpy(
                        np.hstack((
                            np.expand_dims(k, axis=0), 
                            np.expand_dims(input, axis=0)
                            ))
                        ).to(torch.float32)
            dkdt_ = dkdt(k_full)
            return dkdt_[0,:2].detach().numpy()
    
        from scipy.optimize import root

        X_grid, Y_grid = np.meshgrid(
                                np.linspace(xlims[0], xlims[1], int(np.sqrt(num_fp_trials))),
                                np.linspace(ylims[0], ylims[1], int(np.sqrt(num_fp_trials)))
                                )
        fps = []
        for i in range(X_grid.ravel().size):
            x0 = np.array([X_grid.ravel()[i], Y_grid.ravel()[i]])
            #again, assume input equilibriates
            sol = root(DKDT, x0=x0, args=input[trnst,:])
            if all([np.linalg.norm(sol.x - other.x) > dist_thresh for other in fps]):
                fps.append(sol)

        for fp_sol in fps:
            ax.scatter(fp_sol.x[0], fp_sol.x[1], marker="^", c="hotpink")
            
            k = np.hstack(
                        (fp_sol.x,
                         input[trnst,:]
                        )
            )
            """
            gains = compute_deltas(model, k)
            from scipy.optimize._numdiff import approx_derivative
            Jac = approx_derivative(DKDT, k, args=(input,), method="2-point")
            print(Jac)
            #Jac = -np.eye(model.rnn.rank) + np.sum(covs[:nidx, nidx:Iidx,:] * gains, axis=2)
            #eigvals, _= np.linalg.eig(Jac)
            print(eigvals)
            """

    return ax
    
def plot_input_interpolation_lr(model, 
                                inputs, 
                                trnst=50, 
                                fps: Optional[List]=None
                                ):

    fig = plt.figure()
    ax = fig.add_subplot(111,  projection='3d')

    if model.rnn.params["low_rank"]:
        from utils import proj_, get_lr_net_params
        from copy import deepcopy

        m, n, _, _ = get_lr_net_params(model)

        for idx in range(inputs.shape[0]):
            rates, _ = model(inputs[idx,:,:].unsqueeze(dim=0))
            rates = rates.detach().numpy()
            ks = proj_(m.T, rates)
            ax.plot(ks[0,trnst:], ks[1,trnst:], ks[2, trnst:])
            if fps:
                fp = fps[idx].x_star.detach().numpy()
                fpk = proj_(m.T, np.expand_dims(fp, axis=0))
                ax.scatter(fpk[0,0], fpk[1,0], fpk[2,0])

    else:
        pass

    plt.show()



def plot_traj_similarity(model, inputs, base_idx=0, num_angles=NUM_ANGLES, trnst=100):

    """
    Computes and plots similarity of RNN trajectories for different inputs, as
    computed through alignment of SVD components
    """

    import matplotlib.ticker as tck
    num_inputs = inputs.shape[0]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0,1, num_inputs)))
    colors1 = plt.cm.coolwarm(np.linspace(0, 1, inputs.shape[0]))

    num_inputs = inputs.shape[0]
    rates, out = model(inputs[0, :, :].unsqueeze(dim=0))
    ratess = torch.zeros((num_inputs, *rates.shape[1:]))
    ratess[0, :] = rates
    
    for inputs_idx in range(1, num_inputs):
        rates, out = model(inputs[inputs_idx, :, :].unsqueeze(dim=0))
        ratess[inputs_idx, :, :] = rates.squeeze()
    
    #base rates for comparison
    rates_base = ratess[base_idx,trnst:,:].detach().numpy().squeeze()
    U_base, S_base, V_base = np.linalg.svd(rates_base)
    sims = []

    #this is just for the tickering -> TODO: convert to pi
    angles = np.array([np.arccos(inputs[idx,0,0] / np.linalg.norm(inputs[idx,0,0])) for idx in range(inputs.shape[0])])
    angle = angles[0]
    angles = [np.angle(complex(np.cos(angle), np.sin(angle)), deg=True) for angle in angles] 

    calc_m = lambda rates_base, rates, S_base, S: np.linalg.norm(
            rates_base @ rates.T
            ) \
               / np.linalg.norm(
                    np.multiply(S_base,S)
                )
    
    #calculate alignments for all inputs and store 
    for input_idx in range(num_inputs):
        U, S, V = np.linalg.svd(ratess[input_idx,trnst:,:].detach().numpy().squeeze())
        rates_idx = ratess[input_idx,trnst:,:].detach().numpy().squeeze()
        m = calc_m(rates_base, rates_idx, S_base, S)
        sims.append(m)


    #actual plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = sims 
    ax.scatter(angles[0], res[0], color=colors1[0], marker='s', label="reference")
    ax.scatter(angles[1:], res[1:], color=colors1[1:], marker='s')
    #ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $^\circ$'))    
    ax.set_xlabel(r"$\phi$ in deg")
    ax.set_ylabel(r"Sim = $\frac{\left \| X_0 X \right \|_F}{\left \| S_0  S \right \|_F}$", size=20, labelpad=10)
    ax.plot(angles, sims / max(sims), color='black', alpha=.6, zorder=-1)
    #ax.axvline(45, linestyle = '-.', color='black', alpha=.5)
    #ax.axhline(.5, linestyle = '-.', color='black', alpha=.5)
    res = sims / max(sims)
    ax.legend()
    
    return ax

def lower_triangular_heatmap(covs, 
                             eps=0.2, 
                             eps_var=2.0, 
                             figsize=None, 
                             dpi=100, 
                             fontsize=8, 
                             label_fs=6, 
                             numbers_fs=6, 
                             axs=None, 
                             labels=None,
                             put_text=False,
                             **kwargs):
    
    #for text color luminance-control
    import matplotlib.colors as mcolors
    cmap1 = plt.get_cmap('PRGn')
    cmap2 = plt.get_cmap('hot')


    _, num_features, num_pops = covs.shape
    if axs is None:
        fig, axs = plt.subplots(1, num_pops, figsize=figsize, dpi=dpi, squeeze=False)

    for popIdx in range(num_pops):
    
        mat = covs[:,:,popIdx]
        mask = np.zeros_like(mat, dtype=bool)
        mask[np.triu_indices_from(mat)] = True
        mask[np.diag_indices_from(mat)] = False
        tri_mat = np.where(~mask, np.nan, mat)
        tri_mat_flattened = np.where(np.abs(tri_mat)<eps, 0, tri_mat)

        mask_variance = np.eye(mat.shape[0], dtype=bool)
        mask_variance = np.where(~mask_variance, np.nan, mask_variance)
        mask_variance *= mat
        

        if num_pops == 1:
            axs[0,popIdx].xaxis.tick_top()
            axs[0,popIdx].yaxis.tick_right()
            axs[0,popIdx].spines["bottom"].set_visible(False)
            axs[0,popIdx].spines["left"].set_visible(False)

            im = axs[0,popIdx].imshow(tri_mat_flattened, cmap='PRGn')
            norm = im.norm
            imv = axs[0,popIdx].imshow(mask_variance, cmap='hot')
            normv = imv.norm 

            if labels is not None:
                
                axs[0,popIdx].set_xticks(np.arange(len(labels)), labels, fontsize=label_fs)
                axs[0,popIdx].set_yticks(np.arange(len(labels)), labels, fontsize=label_fs)

        else:
            
            axs[0,popIdx].xaxis.tick_top()
            axs[0,popIdx].yaxis.tick_right()
            axs[0,popIdx].spines["bottom"].set_visible(False)
            axs[0,popIdx].spines["left"].set_visible(False)
            axs[0,popIdx].set_title(f'Population {popIdx+1}', pad=12.5)
        

            im = axs[0,popIdx].imshow(tri_mat_flattened, cmap='PRGn')
            norm = im.norm
            imv = axs[0,popIdx].imshow(mask_variance, cmap='hot')
            normv = imv.norm

        if put_text:
            ptile = kwargs.get('ptile', 0)

            for (i,j) in zip(*np.triu_indices_from(covs[:,:,popIdx])):
                #print only most significant values:
                if j > i and np.abs(tri_mat_flattened[i,j]) >= np.percentile(np.abs(covs[:,:, popIdx]), ptile):
                    if np.abs(covs[i,j, popIdx]) > 1e-1:
                        r, g, b, _ = cmap1(norm(covs[i,j,popIdx]))
                        # Compute perceived luminance (human vision weighted)
                        luminance = 0.2126*r + 0.7152*g + 0.0722*b

                        axs[0,popIdx].text(j,i, "{:.1f}".format(tri_mat_flattened[i,j]),
                                           c="white" if luminance < .5 else "black",
                                           ha="center", va="center", fontsize='xx-small'
                                           )

            for i in range(covs.shape[0]):
                r, g, b, _ = cmap2(normv(covs[i,i,popIdx]))
                # Compute perceived luminance (human vision weighted)
                luminance = 0.2126*r + 0.7152*g + 0.0722*b

                axs[0,popIdx].text(i,i, "{:.1f}".format(mask_variance[i,i]),
                                        c="white" if luminance < .5 else "black",
                                        ha="center", va="center", fontsize='xx-small'
                                           )
        if labels is not None:
                
            axs[0,popIdx].set_xticks(np.arange(len(labels)), labels, fontsize=label_fs)
            if popIdx != num_pops-1:
                axs[0,popIdx].set_yticks([], labels=[]) 
            else:
                axs[0,popIdx].set_yticks(np.arange(len(labels)), labels, fontsize=label_fs)
                    
    if num_pops!=1:           
        fig.colorbar(im, ax=axs[:num_pops], shrink=0.4, location='right', pad=0.1)
        fig.colorbar(imv, ax=axs[:num_pops], shrink=0.4, location='bottom', pad=0.1)

    else:

        fig.colorbar(im, ax=axs, shrink=0.4, location='right', pad=0.1)
        fig.colorbar(imv, ax=axs, shrink=0.4, location='bottom', pad=0.1)

def connectivity_scatter(vec1, 
                         vec2, 
                         x: Optional[np.ndarray]=None, 
                         mode ="regular", 
                         ax=None, 
                         **kwargs
                         ):
    

    """
    Scatters neurons in connectivity plane defined by vec1 and vec2

    @Args:
    vec1, vec2: np.ndarray[size]
    x: Fixed point. Gains of neurons at FP act as the weights, same as vec1
    mode: RegMode
        "regular": performs regular regression over the entire population
        "weighted": weighted regression, where the gains act as weights
        "hard_thresh": hard_threshold, populations correspond to neurons in saturating regime (phiPrime > thresh) and linear regime (phiPrime < thresh)

    """

    from sklearn.linear_model import LinearRegression
    from utils import extract_num_pops, phiPrime
    

    pops = kwargs.get('pops', np.zeros(len(vec1)))
    num_pops = extract_num_pops(pops)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    base_colors = prop_cycle.by_key()['color']
    idx = np.linspace(0, len(base_colors)-1, num_pops, dtype=int)
    colors = [base_colors[i] for i in idx]

    if mode not in ["regular", "weighted", "hard_thresh"]:
        raise ValueError('Sorry, need to supply valid regression mode')
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    
    if mode == "regular":
        assert kwargs.get("pops") is not None
        pops = kwargs.get("pops")
        weights = np.ones_like(pops)
        alphas = np.ones_like(pops)*.5

    elif mode == "weighted":
        assert kwargs.get("pops") is not None
        pops = kwargs.get("pops")
        weights = phiPrime(x)[0,:]
        alphas = np.ones_like(x)
        
    elif mode == "hard_thresh":
        thresh = kwargs.get("thresh", .5)
        pops = np.where(phiPrime(x) > thresh, 1, 0)[0]
        weights = pops 
        alphas = phiPrime(x)[0,:]

    xmin, xmax = vec1.min(), vec1.max()
    xvals = np.linspace(xmin, xmax, 100)
    regression = LinearRegression()

    #note vecs need to have appropriate dimensions -> reshape(-1, 1) if passing a 1-d object
    regression.fit(vec1.reshape(-1, 1), vec2.reshape(-1, 1), weights)
    slope, interc = regression.coef_, regression.intercept_
    line_fit_all = ax.plot(xvals, (slope * xvals + interc)[0,:], c="black", linestyle="--", zorder=10000)

    for idx, color in enumerate(colors):

        pt_cloud = ax.scatter(vec1[pops==idx], vec2[pops==idx], color=color, label=f"Population {idx+1}", alpha=alphas, edgecolors='none')
        regression.fit(vec1[pops==idx].reshape(-1, 1), vec2[pops==idx].reshape(-1, 1))
        slope, interc = regression.coef_, regression.intercept_
        line_fit = ax.plot(xvals, (slope * xvals + interc)[0,:], c=color, linestyle="--")


    return ax

def idx2connectivity_params(model, idxs1, idxs2):
    from utils import get_model_conn_vecs
    conn_vecs = get_model_conn_vecs(model)
    vecs1 = conn_vecs[idxs1, :]
    vecs2 = conn_vecs[idxs2, :]
    labels1 = np.array(build_labels(model))[idxs1]
    labels2 = np.array(build_labels(model))[idxs2]

    return vecs1, vecs2, labels1, labels2

def connectivity_scatter_grid(
        vecs1,
        vecs2,
        pops,
        row_labels=[],
        col_labels=[],
        figsize=(13,13),
        spacing=.14,
        bins=50,
        kde_bw="scott"
):
    from scipy.stats import gaussian_kde
    nrows = vecs1.shape[0] 
    ncols = vecs2.shape[0]
    
    cmap = plt.get_cmap(plt.rcParams["image.cmap"])
    colors = cmap(np.linspace(0, 1, extract_num_pops(pops)))

    hists = [dict(
        color=colors[i],
        alpha=.22
        )
        for i in range(extract_num_pops(pops))
        ]

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        nrows + 1,
        ncols + 1,
        figure=fig,
        width_ratios=[1] * ncols + [0.42],
        height_ratios=[0.42] + [1] * nrows,
        wspace=spacing,
        hspace=spacing,
    )

    for i in range(nrows):
        for j in range(ncols):
            
            ax = fig.add_subplot(gs[i+1, j])
            connectivity_scatter(vecs1[i,:], vecs2[j,:], ax=ax, pops=pops)

            if j != 0:
                ax.set_yticks([])

            if i != nrows - 1:
                ax.set_xticks([])

    #Add the marginals
    
    #top row
    for j in range(ncols):

        ax_top = fig.add_subplot(gs[0, j])
        for p in set(pops):
            #hists
            ax_top.hist(vecs2[j,:][pops==p], bins=bins, density=True, **hists[p])

        # KDE fits
        grid = np.linspace(
            min([vecs2[j,:][pops==p].min() for p in pops]),
            max([vecs2[j,:][pops==p].max() for p in pops]),
            300,
        )

        kdes = [gaussian_kde(vecs2[j,:][pops==p], bw_method=kde_bw) for p in set(pops)]
        [ax_top.plot(grid, kde(grid), lw=2, color=colors[idx]) for idx, kde in enumerate(kdes)]
        
        if j != 0:
            ax_top.set_yticks([])

        ax_top.set_xticks([])

        ax_top.xaxis.set_label_position("top")

        ax_top.set_xlabel(
                col_labels[i],
                fontsize=15,
                labelpad=28,
                va="center",
            )


    #right border
    for i in range(nrows):

        ax_right = fig.add_subplot(gs[i+1, ncols])
        for p in set(pops):
            #hists
            ax_right.hist(vecs1[j,:][pops==p], bins=bins, density=True, orientation="horizontal", **hists[p])

        # KDE fits
        grid = np.linspace(
            min([vecs1[j,:][pops==p].min() for p in pops]),
            max([vecs1[j,:][pops==p].max() for p in pops]),
            300,
        )

        kdes = [gaussian_kde(vecs1[j,:][pops==p], bw_method=kde_bw) for p in set(pops)]
        [ax_right.plot(kde(grid), grid, lw=2, color=colors[idx]) for idx, kde in enumerate(kdes)]

        if i != nrows - 1:
            ax_right.set_xticks([])

        ax_right.set_yticks([])

        # row labels on right
        if row_labels is not None:

            ax_right.yaxis.set_label_position("right")

            ax_right.set_ylabel(
                row_labels[i],
                fontsize=15,
                rotation=270,
                labelpad=28,
                va="center",
            )

    plt.show()

def connectivity_scatter_from_idxs(model, idxs1, idxs2, pops): 

    #it's better to have pops seoparately passed, for sanity checks
    vecs1, vecs2, labels1, labels2 = idx2connectivity_params(model, idxs1, idxs2)
    connectivity_scatter_grid(vecs1, vecs2, pops, labels1, labels2)







    










