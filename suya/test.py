#Figure1: (1)Data noises versus time (2) Detection scores versus time
##Take the entire sample that illustrate the changes in both data points and detection scores at the chane point

#Figure2: CADD versus -log(FAR) or \gamma, Here FAR = exp{-\gamma},
##Take various values of \gamma, which is the threshold for detection procedure


import numpy as np
import torch
import matplotlib.pyplot as plt
from models import MVN
from modules import CUSUM, CUSUMF, B_Stat, SRP

from modules.utils import GaussianRBF, Kernel

def load_kernel(name) -> Kernel:
    if name == 'gaussian_rbf':
        kernel = GaussianRBF()
    else:
        raise NotImplementedError(f"No implementation found for: {name}")
    return kernel

def detection_procedure(detector, samples, change_point, null_model, alter_model):
    """
    A sequential detection procedure
    """
    t = 0
    N = len(samples)
    sample_stream = iter(samples)

    detector._reset()
    while (not detector.stop) and t<N:
        t += 1
        new_x = next(sample_stream)
        detector._update(new_x.reshape(1, -1), null_model, alter_model)
    if t>change_point:
        return t-change_point

#Model definition and Data Generation
# mean = torch.tensor([0.]*20)
# alter_mean = torch.tensor([0.31]*20)
# logvar = torch.diag(torch.tensor([1.]*20)).log()

mean = torch.tensor([0.,1.])
logvar = torch.tensor([[0., 1.], [1., 5.]])
ptb_mean = 1 * torch.randn(mean.size())
alter_mean = mean + ptb_mean

change_point = 10
total_length = 10000

null_mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean, logvar.exp())
null_model  = MVN(mean, logvar)
initial_data = null_mvn.sample((1000,))

alter_mvn = torch.distributions.multivariate_normal.MultivariateNormal(alter_mean, logvar.exp())
alter_model  = MVN(alter_mean, logvar)

#calculate lower bound
invcov = torch.linalg.inv(logvar.exp())
t1 = 0.5 * ((alter_mean - mean).matmul(invcov)).matmul((alter_mean - mean).t())
KL_divergence = t1.item()

#plot set up
cm = 1/2.54
width = 40; height = 30
figsize = (width*cm,height*cm)
inputs = np.arange(total_length)

def experiment(hyper_threshold, num_trials, Figure1=True):
    #Load the detector
    hyper_lambda = 1
    cusumf = CUSUMF(hyper_lambda, hyper_threshold)
    cusum = CUSUM(hyper_threshold)
    srp = SRP(hyper_threshold)

    #repeat experiments to calculate empirical EDD
    edds_cusum = []
    edds_cusumf = []
    edds_srp = []
    edds_scanb = []
    for i in range(num_trials):
        null = null_mvn.sample((change_point,))
        alter = alter_mvn.sample((total_length-change_point,))
        sample = torch.cat((null, alter), dim=0)

        scanb = B_Stat(kernel=load_kernel('gaussian_rbf'), initial_data=initial_data, ert=int(np.exp(hyper_threshold)), window_size = 25, test_every_k=1, n_samples=25000)
        
        edd_cusum = detection_procedure(cusum, sample, change_point, null_model, alter_model)
        edds_cusum.append(edd_cusum)

        edd_cusumf = detection_procedure(cusumf, sample, change_point, null_model, alter_model)
        edds_cusumf.append(edd_cusumf)
    
        edd_srp = detection_procedure(srp, sample, change_point, null_model, alter_model)
        edds_srp.append(edd_srp)

        edd_scanb = detection_procedure(scanb, sample, change_point, null_model, alter_model)
        edds_scanb.append(edd_scanb)

        cp_cusum = np.argmin(np.cumsum(np.array(cusum.cum_nll)))
        cp_cusumf = np.argmin(np.cumsum(np.array(cusumf.cum_hst)))
        cp_srp = np.argmin(np.cumsum(np.array(srp.cum_lr)))

        if i==0 and Figure1:
            #Figure 1
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)
            axes[0].set_title(r'Bivariate Gaussian Signals $x_1$; Mean Shift at change-point$=1000$')
            axes[0].plot(inputs, sample[:,0].numpy(), label = r'Signal, $x_1(k)$')
            axes[0].xaxis.set_ticklabels([])
            axes[1].set_title(r'Bivariate Gaussian Signals $x_2$; Mean Shift at change-point$=1000$')
            axes[1].plot(inputs, sample[:,1].numpy(), label = r'Signal, $x_2(k)$')
            axes[1].xaxis.set_ticklabels([])

            axes[2].plot(np.arange(cusum.stopping_time), np.array(cusum.cum_scores), label = r'CUSUM Detection Scores with Stopping time$={}$'.format(cusum.stopping_time))
            axes[2].plot(np.arange(cusumf.stopping_time), np.array(cusumf.cum_scores), label = r'CUSUM-f Detection Scores with Stopping time$={}$'.format(cusumf.stopping_time))
            axes[2].plot(np.arange(srp.stopping_time), np.array(srp.cum_scores), label = r'SRP Detection Scores with Stopping time$={}$'.format(srp.stopping_time))
            axes[2].plot(inputs, np.array([hyper_threshold]*total_length), label = r'Threshold $\tau=100$')
            axes[2].legend(loc=2, prop={'size': 10})
            axes[2].set_title(r'Decision Scores')
            axes[2].xaxis.set_ticklabels([])

            axes[3].plot(np.arange(cusum.stopping_time), np.array(cusum.cum_nll), label = r'CUMSUM $\Lambda[k]$, Detected change point$={}$'.format(cp_cusum+1))
            axes[3].plot(np.arange(cusumf.stopping_time), np.array(cusumf.cum_hst), label = r'CUMSUM-f $S[k]$, Detected change point$={}$'.format(cp_cusumf+1))
            axes[3].plot(np.arange(srp.stopping_time), np.array(srp.cum_lr), label = r'CUMSUM-f $S[k]$, Detected change point$={}$'.format(cp_srp+1))
            axes[3].legend(loc=2, prop={'size': 10})
            axes[3].set_title(r'CUMSUM of Instantaneous Scores')

            plt.savefig('./figure1.pdf', bbox_inches='tight')
            plt.close()
    print(edds_scanb)
    # edds_cusum = [item for item in edds_cusum if item >=0]
    # edds_cusumf = [item for item in edds_cusumf if item >=0]
    # edds_srp = [item for item in edds_srp if item >=0]
    # edds_scanb = [item for item in edds_scanb if item >=0]
    
    return sum(edds_cusum)/len(edds_cusum), sum(edds_cusumf)/len(edds_cusumf), sum(edds_srp)/num_trials, sum(edds_scanb)/len(edds_scanb)

#repeat experiments to get CADD versus gamma
log_gammas = [np.log(2000), np.log(5000), np.log(10000), np.log(20000)]
num_trials = 100
CADD_cumsumf = []
CADD_cumsum = []
CADD_srp = []
CADD_scanb = []

for log_gamma in log_gammas:
    cadd_cusum, cadd_cusumf, cadd_srp, cadd_scanb = experiment(log_gamma, num_trials, Figure1=False)
    CADD_cumsum.append(cadd_cusum)
    CADD_cumsumf.append(cadd_cusumf)
    CADD_srp.append(cadd_srp)
    CADD_scanb.append(cadd_scanb)
    
#Figure 2
plt.plot(np.array(log_gammas), np.array(CADD_cumsum), label = r'CUSUM')
plt.plot(np.array(log_gammas), np.array(CADD_cumsumf), label = r'CUSUM-f')
# plt.plot(np.array(log_gammas), np.array(CADD_srp), label = r'SRP') #some issue for SRP implementation
plt.plot(np.array(log_gammas), np.array(CADD_scanb), label = r'SCAN-B')
plt.plot(np.array(log_gammas), np.array(log_gammas)/KL_divergence, label='Lower bound')
plt.xlabel(r"log $\gamma$")
plt.ylabel(r"$\mathbb{E}_{\nu}[T-\nu|T\geq \nu]$")
plt.legend()
plt.savefig('./figure2.pdf', bbox_inches='tight')
plt.close()


x = np.linspace(-5,5,100)
pdf = np.array(null_model.pdf(torch.tensor(x)).detach().cpu())
gradient = np.array(null_model.pdf(x).detach().cpu())
logpdf = null_model.score(torch.tensor(x))
plt.plot(x, pdf)
plt.plot(x, logpdf)
plt.show()