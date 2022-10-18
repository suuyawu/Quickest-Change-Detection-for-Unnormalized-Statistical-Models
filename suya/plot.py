from cProfile import label
import numpy as np
import numpy.matlib
import torch
import matplotlib.pyplot as plt
from models import MVN


# mean = torch.tensor([0.,1.])
# logvar = torch.tensor([[1., 0.], [0., 1.]]).log()

# x = np.linspace(-10,10,100)
# y = np.linspace(-10,10,100)
# X, Y = np.meshgrid(x, y)
# x_ = X.flatten()
# y_ = Y.flatten()
# xy = np.vstack((x_, y_)).T

# x_tensor = torch.tensor(xy)
class OneDimensionGM():
    def __init__(self, omega, mean, var):
        self.omega = omega
        self.mean = mean
        self.var = var

    def dlnprob(self, x, density = False):
        rep_x = np.matlib.repmat(x, 1, self.omega.shape[0])
        category_prob = np.exp(- (rep_x - self.mean) ** 2 / (2 * self.var)) / (np.sqrt(2 * np.pi * self.var)) * self.omega
        den = np.sum(category_prob, 1)
        num = ((- (rep_x - self.mean) / self.var) * category_prob).sum(1)
        if density:
            return den
        else:
            return np.expand_dims((num / den), 1)

    def MGprob(self, x):
        den = self.dlnprob(x, density = True)
        return np.expand_dims(den, 1)
    def lnprob(self, x):
        den = self.dlnprob(x, density = True)
        return np.log(den)
    def hscore(self, x):
        rep_x = np.matlib.repmat(x, 1, self.omega.shape[0])
        score = 0.5 * (rep_x-self.mean) ** 2 / self.var **2 - 1 / self.var
        return np.sum(score, 1)



# mean = torch.tensor([-10.])
# logvar = torch.tensor([[1.]])

# ptb_mean = 1
# null_model  = MVN(mean, logvar)

x = np.linspace(-15,15,1000)
# x_tensor = torch.tensor(x).reshape(-1,1)

cm = 1/2.54
width = 50; height = 5
figsize = (width*cm,height*cm)
fig1, axes1 = plt.subplots(nrows=1, ncols=5, figsize=figsize)
fig2, axes2 = plt.subplots(nrows=1, ncols=5, figsize=figsize)
fig3, axes3 = plt.subplots(nrows=1, ncols=5, figsize=figsize)

var = np.array([1, 1])
w = np.array([1/3, 2/3])
mean = np.array([-5, 5])

null_model = OneDimensionGM(w, mean, var)
pdf = np.array(np.exp(null_model.lnprob(x.reshape(-1,1))))
logpdf = np.array(null_model.lnprob(x.reshape(-1,1)))
hscore = np.array(null_model.hscore(x.reshape(-1,1)))

var = np.array([1, 1])
w1 = np.array([19/20, 1/20])
mean1 = np.array([-10, 0])

alter_model = OneDimensionGM(w1, mean1, var)
pdf1 = np.array(np.exp(alter_model.lnprob(x.reshape(-1,1))))
logpdf1 = np.array(alter_model.lnprob(x.reshape(-1,1)))
hscore1 = np.array(alter_model.hscore(x.reshape(-1,1)))

axes1[0].plot(x, pdf, label=r'$p(x)$')
axes1[0].plot(x, pdf1, 'r-.', label=r'$p_1(x)$')
axes1[0].set_ylabel('density')
axes1[0].legend(loc=2)

axes2[0].plot(x, -logpdf, label=r'$\mathcal{S}_{L} (X, P)$')
axes2[0].plot(x, -logpdf1, 'r-.', label=r'$\mathcal{S}_{L} (X, P_1)$')
axes2[0].set_ylabel('log scores')
axes2[0].set_ylim([0, 150])
axes2[0].legend(loc=2)

axes3[0].plot(x, hscore, label=r'$\mathcal{S}_{H} (X, P)$')
axes3[0].plot(x, hscore1, 'r-.', label=r'$\mathcal{S}_{H} (X, P_1)$')
axes3[0].set_ylabel('Hyvarinen scores')
axes3[0].set_ylim([0, 450])
axes3[0].legend(loc=2)

var = np.array([1, 1])
w1 = np.array([7/10, 3/10])
mean1 = np.array([-4, 5])

alter_model = OneDimensionGM(w1, mean1, var)
pdf1 = np.array(np.exp(alter_model.lnprob(x.reshape(-1,1))))
logpdf1 = np.array(alter_model.lnprob(x.reshape(-1,1)))
hscore1 = np.array(alter_model.hscore(x.reshape(-1,1)))

axes1[1].plot(x, pdf, label=r'$p(x)$')
axes1[1].plot(x, pdf1, 'r-.', label=r'$p_2(x)$')
axes1[1].legend(loc=2)

axes2[1].plot(x, -logpdf, label=r'$\mathcal{S}_{L} (X, P)$')
axes2[1].plot(x, -logpdf1, 'r-.', label=r'$\mathcal{S}_{L} (X, P_2)$')
axes2[1].set_ylim([0, 150])
axes2[1].legend(loc=2)

axes3[1].plot(x, hscore, label=r'$\mathcal{S}_{H} (X, P)$')
axes3[1].plot(x, hscore1, 'r-.', label=r'$\mathcal{S}_{H} (X, P_2)$')
axes3[1].set_ylim([0, 450])
axes3[1].legend(loc=2)

var = np.array([1, 1])
w1 = np.array([2/5, 3/5])
mean1 = np.array([-3, 6])

alter_model = OneDimensionGM(w1, mean1, var)
pdf1 = np.array(np.exp(alter_model.lnprob(x.reshape(-1,1))))
logpdf1 = np.array(alter_model.lnprob(x.reshape(-1,1)))
hscore1 = np.array(alter_model.hscore(x.reshape(-1,1)))

axes1[2].plot(x, pdf, label=r'$p(x)$')
axes1[2].plot(x, pdf1, 'r-.', label=r'$p_3(x)$')
axes1[2].legend(loc=2)

axes2[2].plot(x, -logpdf, label=r'$\mathcal{S}_{L} (X, P)$')
axes2[2].plot(x, -logpdf1, 'r-.', label=r'$\mathcal{S}_{L} (X, P_3)$')
axes2[2].set_ylim([0, 150])
axes2[2].legend(loc=2)

axes3[2].plot(x, hscore, label=r'$\mathcal{S}_{H} (X, P)$')
axes3[2].plot(x, hscore1, 'r-.', label=r'$\mathcal{S}_{H} (X, P_3)$')
axes3[2].set_ylim([0, 450])
axes3[2].legend(loc=2)

var = np.array([1, 1])
w1 = np.array([3/10, 7/10])
mean1 = np.array([-2, 8])

alter_model = OneDimensionGM(w1, mean1, var)
pdf1 = np.array(np.exp(alter_model.lnprob(x.reshape(-1,1))))
logpdf1 = np.array(alter_model.lnprob(x.reshape(-1,1)))
hscore1 = np.array(alter_model.hscore(x.reshape(-1,1)))

axes1[3].plot(x, pdf, label=r'$p(x)$')
axes1[3].plot(x, pdf1, 'r-.', label=r'$p_4(x)$')
axes1[3].legend(loc=2)

axes2[3].plot(x, -logpdf, label=r'$\mathcal{S}_{L} (X, P)$')
axes2[3].plot(x, -logpdf1, 'r-.', label=r'$\mathcal{S}_{L} (X, P_4)$')
axes2[3].set_ylim([0, 150])
axes2[3].legend(loc=2)

axes3[3].plot(x, hscore, label=r'$\mathcal{S}_{H} (X, P)$')
axes3[3].plot(x, hscore1, 'r-.', label=r'$\mathcal{S}_{H} (X, P_4)$')
axes3[3].set_ylim([0, 450])
axes3[3].legend(loc=2)

var = np.array([1, 1])
w1 = np.array([1/20, 19/20])
mean1 = np.array([0, 10])

alter_model = OneDimensionGM(w1, mean1, var)
pdf1 = np.array(np.exp(alter_model.lnprob(x.reshape(-1,1))))
logpdf1 = np.array(alter_model.lnprob(x.reshape(-1,1)))
hscore1 = np.array(alter_model.hscore(x.reshape(-1,1)))

axes1[4].plot(x, pdf, label=r'$p(x)$')
axes1[4].plot(x, pdf1, 'r-.', label=r'$p_5(x)$')
axes1[4].legend(loc=2)

axes2[4].plot(x, -logpdf, label=r'$\mathcal{S}_{L} (X, P)$')
axes2[4].plot(x, -logpdf1, 'r-.', label=r'$\mathcal{S}_{L} (X, P_5)$')
axes2[4].set_ylim([0, 150])
axes2[4].legend(loc=2)

axes3[4].plot(x, hscore, label=r'$\mathcal{S}_{H} (X, P)$')
axes3[4].plot(x, hscore1, 'r-.', label=r'$\mathcal{S}_{H} (X, P_5)$')
axes3[4].set_ylim([0, 450])
axes3[4].legend(loc=2)

fig1.savefig('./density.pdf', bbox_inches='tight')
fig2.savefig('./logscore.pdf', bbox_inches='tight')
fig3.savefig('./hscore.pdf', bbox_inches='tight')