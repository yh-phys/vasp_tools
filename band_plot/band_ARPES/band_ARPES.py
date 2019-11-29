#!/usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.use('Agg')  # silent mode

#------------------- rc.Params 1----------------------
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

font = {'family' : 'Times New Roman', 
    'color': 'black',
    'weight': 'normal',
    'size': 13.0,
    }

fig = plt.figure(figsize=(5, 4))
axe = plt.subplot(111)
datas=np.loadtxt('BAND.dat', dtype=np.float64)
group_labels=[]
xtick=[]
with open('KLABELS', 'r') as reader:
    lines=reader.readlines()[1:]
for i in lines:
    s=i.encode('utf-8')  #.decode('latin-1')
    if len(s.split())==2 and not s.decode('utf-8', 'ignore').startswith('*'):
        group_labels.append(s.decode('utf-8', 'ignore').split()[0])
        xtick.append(float(s.split()[1]))
for index in range(len(group_labels)):
    if group_labels[index]=='GAMMA':
        group_labels[index]=u'Γ'

kpt=datas[:,0]
energy=datas[:,1]
ikpt = np.arange(min(kpt), max(kpt)*1.00, 0.01)
ienergy = np.arange(min(energy), max(energy)*1.01, 0.01)
WGHTRS=np.ones(len(energy))
Y, X = np.meshgrid(ikpt, ienergy)   # 303, 28
iweight=np.zeros((len(ienergy),len(ikpt)))
sK = 0.02;  # smearing factor in k-space
sE = 0.02;  # smearing factor in energy
def smoothing(e,E):
        return 1./((e-E)**3 + 2.1)

def gauss(X, x0, s):
    return 1/(s*np.sqrt(2*np.pi)) * np.exp(-((X-x0)**2)/(2*s**2));

for l in range(len(energy)):
      for i in range(len(ienergy)):
            if (abs(ienergy[i]-energy[l]) > sK):
               continue
            for j in range(len(ikpt)):
                  if (abs(ikpt[j]-kpt[l]) > sE):
                      continue
                  #wK=smoothing(ienergy[i],energy[l])
                  #wE=smoothing(ikpt[j],kpt[l])
                  wK=gauss(ienergy[i],energy[l],5*sK)
                  wE=gauss(ikpt[j],kpt[l],5*sE)
                  iweight[i,j]=iweight[i,j]+wK*wE
z = iweight[:-1, :-1]
iweight=iweight/(gauss(0,0,sK)*gauss(0,0,sE))
axe.pcolor(Y, X, iweight, vmin=0, vmax=0.1)
plt.ylim((-4,  4))  # set y limits manually
axe.yaxis.set_minor_locator(ticker.MultipleLocator(1.00))  # determine the minor locator of y-axis
axe.yaxis.set_major_locator(ticker.MultipleLocator(2.00))  # determine the major loctor of y-axis
axe.set_xlim((xtick[0], xtick[-1]))
#axe.set_ylabel(r'$\mathrm{E}$-$\mathrm{E_{VBM}}$ (eV)',fontdict=font)
axe.set_ylabel('Energy (eV)', fontdict=font)
#axe.set_xlabel(r'Wavevector',fontdict=font)
axe.set_xticks(xtick)
axe.set_xticklabels(group_labels, rotation=0, fontsize=font['size']-2, fontname=font['family'])
for i in xtick[1:-1]:
    axe.axvline(x=i, ymin=0, ymax=1, linestyle='--', linewidth=0.5, color='0.5')
#axe.colorbar()
#plt.show()
plt.savefig('band.png', bbox='tight', pad_inches=0.1, dpi= 300)
