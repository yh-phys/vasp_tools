#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@author: V. Wang, Jin-Cheng Liu, Nxu
@file: pband_plot.py
@time: 2018/12/18 20:57
A script to plot PBAND
"""

import numpy as np
import matplotlib as mpl
import os
mpl.use('Agg')  # silent mode
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import sys


#------------------- Data Read ----------------------

def getelementinfo():
    try:
        with open("POSCAR",'r') as reader:
            line_s=reader.readlines()
    except:
        print("No POSCAR found!")
    try:    
        element_s=line_s[5].rstrip('\r\n').rstrip('\n')
        elements=element_s.split()
    except:
        print("POSCAR element line is wrong!")
    
    
    data={}
    for i in range(len(elements)):
        if os.path.exists("PBAND." + elements[i] + ".dat"):
           data[elements[i]]=np.loadtxt("PBAND." + elements[i] + ".dat")
        elif os.path.exists("PBAND_" + elements[i] + ".dat"):
           data[elements[i]]=np.loadtxt("PBAND_" + elements[i] + ".dat")
        
    return data,elements
def getelementinfo_spin():
    try:
        with open("POSCAR",'r') as reader:
            line_s=reader.readlines()
    except:
        print("No POSCAR found!")
    try:    
        element_s=line_s[5].rstrip('\r\n').rstrip('\n')
        elements=element_s.split()
    except:
        print("POSCAR element line is wrong!")
    
    data_up = {}
    data_dw = {}
    
    for i in range(len(elements)):
        if os.path.exists("PBAND_" + elements[i] + "_DW.dat"):
           data_dw[elements[i]]=np.loadtxt("PBAND_" + elements[i] + "_DW.dat")
        if os.path.exists("PBAND_" + elements[i] + "_UP.dat"):
           data_up[elements[i]]=np.loadtxt("PBAND_" + elements[i] + "_UP.dat")
        
    return data_up,data_dw,elements

def getHighSymmetryPoints():
    hsp = np.loadtxt("KLABELS", dtype=np.string_, skiprows=1, usecols=(0, 1))
    group_labels = hsp[:-1, 0].tolist()
    group_labels = [i.decode('utf-8', 'ignore') for i in group_labels]
    for index in range(len(group_labels)):
        if group_labels[index] == "GAMMA":
            group_labels[index] = u"Γ"
    return group_labels, hsp

def maxminnorm(a):
    amin, amax = a.min(), a.max()  # fin maximum minimum
    if amax == 0:
        return a
    else:
        a = (a - amin) / (amax - amin)  # (value-minimum)/(maximum-minimum)
        return a

def getPbandData(data,scaler):
    kpt = data[:, 0]  # kpath
    eng = data[:, 1]  # energy level
    wgt_s = data[:, 2] * scaler  # weight, 20 is enlargement factor
    #wgt_s = maxminnorm(wgt_s) * scaler  # Normlized

    wgt_py = data[:, 3] * scaler  # weight, 20 is enlargement factor
    #wgt_py = maxminnorm(wgt_py)*scaler
    wgt_pz = data[:, 4] * scaler  # weight, 20 is enlargement factor
    #wgt_pz = maxminnorm(wgt_pz)*scaler

    wgt_px = data[:, 5] * scaler  # weight, 20 is enlargement factor
    #wgt_px = maxminnorm(wgt_px)*scaler

    wgt_p = np.array(wgt_py) + np.array(wgt_px) + np.array(wgt_pz)
    #wgt_p = maxminnorm(wgt_p) * scaler  # Normlized


    wgt_dxy = data[:, 6] * scaler
    #wgt_dxy = maxminnorm(wgt_dxy) * scaler  # Normlized

    wgt_dyz = data[:, 7] * scaler
    #wgt_dyz = maxminnorm(wgt_dyz) * scaler
    wgt_dz2 = data[:, 8] * scaler
    #wgt_dz2 = maxminnorm(wgt_dz2) * scaler
    wgt_dxz = data[:, 9] * scaler
    #wgt_dxz = maxminnorm(wgt_dxz) * scaler
    wgt_dx2y2 = data[:, 10] * scaler
    #wgt_dx2y2 = maxminnorm(wgt_dx2y2) * scaler
    wgt_d = np.array(wgt_dxy) + np.array(wgt_dyz) + np.array(wgt_dz2) \
             + np.array(wgt_dxz) + np.array(wgt_dx2y2)
    #wgt_d = maxminnorm(wgt_d) * scaler  # Normlized

    #wgt_tot = maxminnorm(data[:, 11]) * scaler
    wgt_tot = data[:, 11] * scaler
    return kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy,  \
            wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot

#------------------- Pband Plot ----------------------

class pbandplots(object):
    def __init__(self,lwd,op,scaler,energy_limits,font,dpi,figsize,corlor0):
        from matplotlib import pyplot as plt
        self.data,self.elements=getelementinfo()
        self.data_up,self.data_dw,self.elements_spin=getelementinfo_spin()    #2019.8.23
        self.group_labels, self.hsp = getHighSymmetryPoints()    # HighSymmetryPoints_labels 
        self.x = [float(i) for i in self.hsp[:-1, 1].tolist()]   # HighSymmetryPoints_coordinate
        self.lwd=lwd ; self.op=op;self.scaler=scaler;self.energy_limits=energy_limits
        self.font=font;self.dpi=dpi;self.figsize=figsize
        self.corlor0=corlor0 
    def plotfigure(self,ax, kpt, eng, title):
        import matplotlib
        ax.plot(kpt, eng, color=self.corlor0, lw=self.lwd, linestyle='-', alpha=1)
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_ylim(self.energy_limits)
        ytick = np.arange(self.energy_limits[0], self.energy_limits[1], 2)
        a = int(len(ytick) / 2)
        plt.yticks(np.insert(ytick, a, 0))
        ax.set_xticks(self.x)
        plt.yticks(fontsize=self.font['size'],fontname=self.font['family'])
        plt.ylabel(r'$Energy$ (eV)',fontdict=self.font,labelpad=-2)
        ax.spines['left'].set_linewidth(0.3)
        ax.spines['right'].set_linewidth(0.3)
        ax.spines['top'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.spines['left'].set_linewidth(0.3)
        ax.spines['right'].set_linewidth(0.3)
        ax.spines['top'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.tick_params(axis='x',pad=1,width=0.5,length=1.5,direction='in')
        ax.tick_params(axis='y',pad=1,width=0.5,length=1.5,direction='in')
        plt.suptitle(title,fontsize=4)
        ax.set_xticklabels(self.group_labels, rotation=0, fontsize=self.font['size'],fontname=self.font['family'])
        ax.axhline(y=0, xmin=0, xmax=1, linestyle='--', linewidth=0.5, color='0.5')
        for i in self.x[1:-1]:
            ax.axvline(x=i, ymin=0, ymax=1, linestyle='--', linewidth=0.5, color='0.5')
        ax.set_xlim((self.x[0], self.x[-1]))
        return plt
    def plotfigure_dw(self,ax, kpt, eng, title):
        import matplotlib
        ax.plot(kpt, eng, color=self.corlor0, lw=self.lwd, linestyle='-', alpha=1)
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_ylim(self.energy_limits)
        ytick = np.arange(self.energy_limits[0], self.energy_limits[1], 2)
        a = int(len(ytick) / 2)
        #plt.yticks(np.insert(ytick, a, 0))
        ax.set_xticks(self.x)
        plt.yticks([])
        #plt.yticks(fontsize=self.font['size'],fontname=self.font['family'])
        #plt.ylabel(r'$Energy$ (eV)',fontdict=self.font,labelpad=-2)
        ax.spines['left'].set_linewidth(0.3)
        ax.spines['right'].set_linewidth(0.3)
        ax.spines['top'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.spines['left'].set_linewidth(0.3)
        ax.spines['right'].set_linewidth(0.3)
        ax.spines['top'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.tick_params(axis='x',pad=1,width=0.5,length=1.5,direction='in')
        #ax.tick_params(axis='y',pad=1,width=0.5,length=1.5,direction='in')
        plt.suptitle(title,fontsize=4)
        ax.set_xticklabels(self.group_labels, rotation=0, fontsize=self.font['size'],fontname=self.font['family'])
        ax.axhline(y=0, xmin=0, xmax=1, linestyle='--', linewidth=0.5, color='0.5')
        for i in self.x[1:-1]:
            ax.axvline(x=i, ymin=0, ymax=1, linestyle='--', linewidth=0.5, color='0.5')
        ax.set_xlim((self.x[0], self.x[-1]))
        return plt
    def plotPbandAllElementsspd(self):
        from matplotlib import pyplot as plt
        print("start plot PBAND for all Elements with s p d projection in one figure !...")
        colorcode = ['blue', 'cyan', 'red', 'green', 'yellow','purple','chartreuse','fuchsia','orangered','hotpink','violet','teal']  #if number of orbitals are more 3,one need increase the number of colors in "colorcode"
        markerorder=['o','v','p','*','>','s','1','2','3','4','x','+']
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        legend_s=[]
        for elementorder in range(len(self.elements)):
            kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot \
            = getPbandData(self.data[self.elements[elementorder]],self.scaler)
            if (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[elementorder], edgecolor=colorcode[elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[elementorder])
               legend_s.append('$' + self.elements[elementorder] + '$'+'_s')
               continue
            elif (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[2*elementorder], edgecolor=colorcode[2*elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[2*elementorder])
               ax.scatter(kpt, eng, s=wgt_p,color=colorcode[2*elementorder+1], edgecolor=colorcode[2*elementorder+1], linewidths=0.2,\
               alpha=op*0.5,marker=markerorder[2*elementorder+1])
               legend_s.append('$' + self.elements[elementorder] + '$'+'_s')
               legend_s.append('$' + self.elements[elementorder] + '$'+'_p')
               continue
            elif not (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[3*elementorder], edgecolor=colorcode[3*elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[3*elementorder])
               ax.scatter(kpt, eng, s=wgt_p,color=colorcode[3*elementorder+1], edgecolor=colorcode[3*elementorder+1], linewidths=0.2,\
               alpha=op*0.5,marker=markerorder[3*elementorder+1])
               ax.scatter(kpt, eng, s=wgt_d, color=colorcode[3*elementorder+2], edgecolor=colorcode[3*elementorder+2],linewidths=0.2,\
               alpha=op*0.4,marker=markerorder[3*elementorder+2])
               legend_s.append('$' + self.elements[elementorder] + '$'+'_s')
               legend_s.append('$' + self.elements[elementorder] + '$'+'_p')
               legend_s.append('$' + self.elements[elementorder] + '$'+'_d')
               continue
       
        ax.legend(tuple(legend_s),loc='best',fontsize=5, shadow=False, labelspacing=0.1,framealpha=0.2,handlelength=0.5,handleheight=0.1)
        title0=" "
        for atom in range(len(self.elements)):
            title0=self.elements[atom] + title0
        plt = self.plotfigure(ax, kpt, eng, "        AlBi" )

        plt.subplots_adjust(top=0.950,bottom=0.05,left=0.15,right=0.99,wspace=0)
        plt.savefig('PBND'+title0.rstrip('\r\n').rstrip()+'spd.eps',img_format=u'eps', dpi=1000)
             #del ax, fig
    def plotPbandAllElementsspd_spin(self):
        from matplotlib import pyplot as plt
        print("start plot PBAND with spin calculation for all Elements with s p d projection in one figure !...")
        colorcode = ['blue', 'cyan', 'red', 'green', 'yellow','purple','chartreuse','fuchsia','orangered','hotpink','violet','teal']  #if number of orbitals are more 3,one need increase the number of colors in "colorcode"
        markerorder=['o','v','p','*','>','s','1','2','3','4','x','+']
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(121)
        legend_s=[]
        for elementorder in range(len(self.elements_spin)):
            kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot \
            = getPbandData(self.data_up[self.elements_spin[elementorder]],self.scaler)
            if (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[elementorder], edgecolor=colorcode[elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[elementorder])
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_s')
               continue
            elif (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[2*elementorder], edgecolor=colorcode[2*elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[2*elementorder])
               ax.scatter(kpt, eng, s=wgt_p,color=colorcode[2*elementorder+1], edgecolor=colorcode[2*elementorder+1], linewidths=0.2,\
               alpha=op*0.5,marker=markerorder[2*elementorder+1])
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_s')
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_p')
               continue
            elif not (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[3*elementorder], edgecolor=colorcode[3*elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[3*elementorder])
               ax.scatter(kpt, eng, s=wgt_p,color=colorcode[3*elementorder+1], edgecolor=colorcode[3*elementorder+1], linewidths=0.2,\
               alpha=op*0.5,marker=markerorder[3*elementorder+1])
               ax.scatter(kpt, eng, s=wgt_d, color=colorcode[3*elementorder+2], edgecolor=colorcode[3*elementorder+2],linewidths=0.2,\
               alpha=op*0.4,marker=markerorder[3*elementorder+2])
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_s')
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_p')
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_d')
               continue
            
        #ax.legend(tuple(legend_s),loc='best',fontsize=15, shadow=False, labelspacing=0.1,framealpha=0.2,handlelength=0.5,handleheight=0.1)
        title0=" "
        for atom in range(len(self.elements)):
            title0=self.elements_spin[atom] + title0
        plt = self.plotfigure(ax, kpt, eng, "" )
        ax = fig.add_subplot(122)
        legend_s=[]
        for elementorder in range(len(self.elements_spin)):
            kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot \
            = getPbandData(self.data_dw[self.elements_spin[elementorder]],self.scaler)
            if (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[elementorder], edgecolor=colorcode[elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[elementorder])
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_s')
               continue
            elif (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[2*elementorder], edgecolor=colorcode[2*elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[2*elementorder])
               ax.scatter(kpt, eng, s=wgt_p,color=colorcode[2*elementorder+1], edgecolor=colorcode[2*elementorder+1], linewidths=0.2,\
               alpha=op*0.5,marker=markerorder[2*elementorder+1])
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_s')
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_p')
               continue
            elif not (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
               ax.scatter(kpt, eng, s=wgt_s, color=colorcode[3*elementorder], edgecolor=colorcode[3*elementorder], linewidths=0.2,\
               alpha=op*0.7,marker=markerorder[3*elementorder])
               ax.scatter(kpt, eng, s=wgt_p,color=colorcode[3*elementorder+1], edgecolor=colorcode[3*elementorder+1], linewidths=0.2,\
               alpha=op*0.5,marker=markerorder[3*elementorder+1])
               ax.scatter(kpt, eng, s=wgt_d, color=colorcode[3*elementorder+2], edgecolor=colorcode[3*elementorder+2],linewidths=0.2,\
               alpha=op*0.4,marker=markerorder[3*elementorder+2])
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_s')
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_p')
               legend_s.append('$' + self.elements_spin[elementorder] + '$'+'_d')
               continue
            
        ax.legend(tuple(legend_s),bbox_to_anchor=(1.01,0),loc=3,fontsize=12, \
		shadow=False, labelspacing=0.1,framealpha=0.2,handlelength=2,handleheight=2)
        title0=" "
        for atom in range(len(self.elements)):
            title0=self.elements[atom] + title0
        plt = self.plotfigure_dw(ax, kpt, eng, "  " )
        plt.subplots_adjust(top=0.950,bottom=0.05,left=0.07,right=0.75,wspace=0)
        plt.savefig('PBND'+title0.rstrip('\r\n').rstrip()+'spd.png',img_format=u'png', dpi=1000)
             #del ax, fig    
    def plotPbandspd(self):
        from matplotlib import pyplot as plt
        print("start plot PBAND for each Elements with s p d projection...")
        for element in self.elements:
            print("plot ", element)
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111)
            kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot\
                = getPbandData(self.data[element],self.scaler)
            ax.scatter(kpt, eng, wgt_s, color='blue', edgecolor='blue', linewidths=0.2, alpha=self.op,marker='o')
            ax.scatter(kpt, eng, wgt_p, color='cyan', edgecolor='cyan', linewidths=0.2, alpha=self.op - 0.6,marker='v')
            ax.scatter(kpt, eng, wgt_d, color='red', edgecolor='red', linewidths=0.2, alpha=self.op - 0.85,marker='*')

            if (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
                ax.legend(('$s$'), loc='best', shadow=False, labelspacing=0.1)
            elif (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
                ax.legend(('$s$','$p$'), loc='best', shadow=False, labelspacing=0.1)
            elif not (np.array(wgt_d) == np.zeros_like(np.array(wgt_d))).all() and not (np.array(wgt_p) == np.zeros_like(np.array(wgt_p))).all():
                ax.legend(('$s$','$p$','$d$'), loc='best', shadow=False, labelspacing=0.1)
            plt = self.plotfigure(ax, kpt, eng, element)
            plt.subplots_adjust(top=0.950,bottom=0.110,left=0.1,right=0.99,wspace=0)
            plt.savefig('PBAND' + element + 'spd.png',bbox_inches='tight',pad_inches=0.1, dpi=self.dpi)
            #del ax, fig

    def plotPbandAllElements(self):
        from matplotlib import pyplot as plt
        print("start plot PBAND including Elements...")
        colorcode = ['blue', 'cyan', 'red', 'green', 'yellow']
        markerorder=['o','v','p','*','>']
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        for elementorder in range(len(self.elements)):
            #del wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot
            kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot \
                = getPbandData(self.data[self.elements[elementorder]],self.scaler)
            ax.scatter(kpt, eng, wgt_tot, color=colorcode[elementorder], edgecolor=colorcode[elementorder], \
                       linewidths=0.2, alpha=self.op-elementorder*0.2,marker=markerorder[elementorder])

        if len(self.elements) == 5:
            ax.legend(('$' + self.elements[0] + '$', '$' + self.elements[1] + '$', '$' + self.elements[2] + '$', '$' + self.elements[3] + '$', '$' + self.elements[4] + '$'),\
                      loc='best', shadow=False, labelspacing=0.1)
        elif len(self.elements) == 4:
            ax.legend(('$' + self.elements[0] + '$', '$' + self.elements[1] + '$', '$' + self.elements[2] + '$', '$' + self.elements[3] + '$'),\
                      loc='best', shadow=False, labelspacing=0.1)
        elif len(self.elements) == 3:
            ax.legend(('$' + self.elements[0]+'$', '$' + self.elements[1] + '$', '$' + self.elements[2] + '$'),\
                      loc='best', shadow=False, labelspacing=0.1)
        elif len(self.elements) == 2:
            ax.legend(('$' + self.elements[0] + '$', '$' + self.elements[1] + '$'), \
                      loc='best', shadow=False, labelspacing=0.1)
        elif len(self.elements) == 1:
            ax.legend(('$' + self.elements[0] + '$'), \
                      loc='best', shadow=False, labelspacing=0.1)
        title0=" "
        for atom in range(len(self.elements)):
            title0=self.elements[atom] + title0 
        plt = self.plotfigure(ax, kpt, eng, title0)
        #  plt = self.plotfigure(ax, kpt, eng, self.elements[elementorder])
        plt.subplots_adjust(top=0.850,bottom=0.110,left=0.165,right=0.855,wspace=0)
        plt.savefig('PBAND.png',bbox_inches='tight',pad_inches=0.1, dpi=self.dpi)
        #del ax, fig

    def plotPbandEachElements(self):
        from matplotlib import pyplot as plt
        print("start plot PBAND including each Elements...")
        colorcode = ['red','cyan','blue' , 'green', 'yellow']

        for elementorder in range(len(self.elements)):
            print("plot ", self.elements[elementorder])
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111)
            #del wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot
            kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot \
                = getPbandData(self.data[self.elements[elementorder]],self.scaler)
            ax.scatter(kpt, eng, wgt_tot, color=colorcode[elementorder], edgecolor=colorcode[elementorder], \
                       linewidths=0.2, alpha=self.op-elementorder*0.2,marker='v')
            #print(elements)
            #ax.legend(elements[elementorder], shadow=False, labelspacing=0.1)
            plt.legend(labels=['$' + self.elements[elementorder] + '$'],shadow=False, labelspacing=0.1,loc='best')
            plt = self.plotfigure(ax, kpt, eng, self.elements[elementorder])
            plt.subplots_adjust(top=0.950,bottom=0.110,left=0.165,right=0.855,wspace=0)
            plt.savefig('PBAND'+self.elements[elementorder]+'.eps',bbox_inches='tight',img_format=u'eps',pad_inches=0.1, dpi=self.dpi)
            #del ax, fig

    def plotPbandpxpypz(self):
        from matplotlib import pyplot as plt
        print("start plot PBAND including s pxpypz for each Elements...")
        colorcode = ['blue', 'cyan', 'red', 'green', 'yellow']

        for elementorder in range(len(self.elements)):
            print("plot ", self.elements[elementorder])
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111)
            #del wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot
            kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot \
                = getPbandData(self.data[self.elements[elementorder]],self.scaler)
            ax.scatter(kpt, eng, wgt_s, color='blue', edgecolor='blue', alpha=self.op,marker='o')
            ax.scatter(kpt, eng, wgt_py, color='cyan', edgecolor='cyan', alpha=self.op - 0.1,marker='v')
            ax.scatter(kpt, eng, wgt_pz, color='red', edgecolor='red', alpha=self.op - 0.4,marker='p')
            ax.scatter(kpt, eng, wgt_px, color='magenta', edgecolor='magenta', alpha=self.op - 0.7,marker='*')
            ax.legend(('$s$', '$p_y$', '$p_z$', '$p_x$'), loc='upper right', shadow=False, labelspacing=0.1)

            plt = self.plotfigure(ax, kpt, eng, self.elements[elementorder])
            plt.subplots_adjust(top=0.950,bottom=0.110,left=0.165,right=0.855,wspace=0)
            plt.savefig('PBAND'+self.elements[elementorder]+'spxpypz.png',bbox_inches='tight',pad_inches=0.1,dpi=self.dpi)
            #del ax, fig
    def plottotalBands(self):
        from matplotlib import pyplot as plt
        print("start plot total BANDs ...")
        colorcode = ['blue', 'cyan', 'red', 'green', 'yellow']
        markerorder=['o','v','p','*','>']
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        kpt, eng, wgt_s, wgt_py, wgt_pz, wgt_px, wgt_p, wgt_dxy, wgt_dyz, wgt_dz2, wgt_dxz, wgt_dx2y2, wgt_d, wgt_tot \
        = getPbandData(self.data[self.elements[0]],self.scaler) 
        title0=" "
        for atom in range(len(self.elements)):
            title0=self.elements[atom] + title0 
        plt = self.plotfigure(ax, kpt, eng, title0)
        #  plt = self.plotfigure(ax, kpt, eng, self.elements[elementorder])
        plt.subplots_adjust(top=0.950,bottom=0.110,left=0.1,right=0.855,wspace=0)
        plt.savefig('BAND.png',bbox_inches='tight',pad_inches=0.1, dpi=self.dpi)
        #del ax, fig

if __name__ == "__main__":
    
    #___________________________________SETUP____________________________________
    
        
    print("    ****************************************************************")
    print("    * This is a code used to plot kinds of bandstructure,written by*")
    print("    *   V.Wang,Jin-Cheng Liu,modfied by Nan Xu,Xue Fei Liu         *")
    print("    ****************************************************************")
    print( "\n")
    print("    ****************************************************************")
    print("    *     Type of bandstructures are obtained as below:            *") 
    print("    * 1).For total bandstructure of all atoms in one figure        *")
    print("    * 2).For projected bands of each element in separated figures  *")    
    print("    * 3).For total spd orbitals of total elements in one figure    *")
    print("    * 4).For s-pxpypz of each element in separated figures         *")
    print("    * 5).For all elements and correspond spd orbitals in one figure*")
    print("    * 6).For a total bandstructure                                 *")
    print("    * 7).For all elements and correspond spd with spin orbitals in one figure*")
    print("    ****************************************************************")
    print("                       (^o^)GOOD LUCK!(^o^)                         ")
    print( "\n")
    
    print( " Band plot initialization... ")
    print( "*******************************************************************")
    print("Please set the color and width of line in figure,input line=0.2")
    print(" and color = 'black' for choice 1->5,input line >= 1 and color =")
    print(" 'red','blue' or .... for choice 6")
    print( "********************************************************************")

    corlor0 = str(input("Input color = ? according to your choice number:"))
    lwd = float(input("Input line =? according to your choice number:"))
    print("*********************************************************************")
    print(  "Which kind of bandstructure do you need plot ?")
    print(  "Please type in a number to select a function: e.g.1, 2 ....,")
    print("*********************************************************************")

    
    op = 1  # Max alpha blending value, between 0 (transparent) and 1 (opaque).
    scaler = 60  # Scale factor for projected band
    energy_limits=(-6, 6.01)  # energy ranges for PBAND
    dpi=1000          # figure_resolution
    figsize=(5, 4)   #figure_inches
    font = {'family' : 'Times New Roman', 
        'color'  : 'black',
        'weight' : 'normal',
        'size' : 10,
        }       #FONT_setup
    pband_plots=pbandplots(lwd,op,scaler,energy_limits,font,dpi,figsize,corlor0)
    try:
        bandtype = int(input("Input number--->"))
    except ValueError:
        raise ValueError(" You have input wrong ! Please rerun this code !")
    

    if bandtype == 1:
        pband_plots.plotPbandAllElements()  # plot pband for all element in one figure
    elif bandtype == 2:
        pband_plots.plotPbandEachElements()  # plot pband for each element
    elif bandtype == 3:
        pband_plots.plotPbandspd()  # plot pband with different angular momentum        
    elif bandtype == 4:
        pband_plots.plotPbandpxpypz()  # plot pband with Magnetic angular momentum   
    elif bandtype == 5:
        pband_plots.plotPbandAllElementsspd() #plot pband of all enlenments and spd orbitals
    elif bandtype == 6:
        pband_plots.plottotalBands()
    elif bandtype == 7:
	    pband_plots.plotPbandAllElementsspd_spin()
    else :
        print(" You have input wrong ! Please rerun this code !" )
        sys.exit(0)     









