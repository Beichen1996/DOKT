import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.font_manager as font_manager
from sys import argv

from matplotlib.axes import Axes
import numpy as np
from PIL import Image
from subprocess import call
from collections import OrderedDict
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import MultipleLocator



def plt_props():
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.style'] = 'normal'
    plt.rcParams['font.variant'] = 'normal' 
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['figure.figsize'] = (6.2, 6.2)
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 5.2

m={}

m['DOKT']='o'
m['Random']='>'
m['ALFA']='P'
m['PT4AL']='^'
m['PAL']='D'
m['LL']='X'




BLEU4_ratios = [item for item in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]

BLEU4_DOKT = [72.51,77.29,78.64,79.20,79.47,79.62,80.01,80.12,80.18,80.26  ]
BLEU4_DOKT_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_LL = [72.51,76.62,78.00,78.59,78.94,79.11,79.07,79.22,79.29,79.71]
BLEU4_LL_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_ALFA = [72.51,76.91,78.30,78.79,79.00,79.21,79.47,79.62,80.01,79.59 ]
BLEU4_ALFA_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_PT4AL = [72.51,74.52,76.29,76.89,77.03,77.47,77.92,78.08,78.17,78.29  ]
BLEU4_PT4AL_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_PAL = [72.51,74.31,76.13,76.81,77.00,77.45,77.89,78.01,78.02,78.12 ]
BLEU4_PAL_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_random = [72.51,74.02,74.80,75.67,76.42,76.75,77.04,77.55,77.58,77.92 ]
BLEU4_random_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0] 

BLEU4_ratios = BLEU4_ratios[1:];BLEU4_DOKT = BLEU4_DOKT[1:];BLEU4_LL = BLEU4_LL[1:];BLEU4_ALFA = BLEU4_ALFA[1:];BLEU4_PT4AL = BLEU4_PT4AL[1:];BLEU4_PAL = BLEU4_PAL[1:];BLEU4_random = BLEU4_random[1:]
fig, ax = plt.subplots(dpi=500, figsize=(6,6))
plt_props()
#plt.rc('font',family='Times New Roman') 
BLEU4_DOKT_plot = plt.errorbar(BLEU4_ratios, BLEU4_DOKT, label='DOKT', marker=m['DOKT'] )
BLEU4_LL_plot = plt.errorbar(BLEU4_ratios, BLEU4_LL, label='LL', marker=m['LL'] )
BLEU4_ALFA_plot = plt.errorbar(BLEU4_ratios, BLEU4_ALFA, label='ALFA', marker=m['ALFA'] )
BLEU4_PT4AL_plot = plt.errorbar(BLEU4_ratios, BLEU4_PT4AL, label='PT4AL', marker=m['PT4AL'])
BLEU4_PAL_plot = plt.errorbar(BLEU4_ratios, BLEU4_PAL, label='PAL', marker=m['PAL'])
BLEU4_random_plot = plt.errorbar(BLEU4_ratios, BLEU4_random, label='Random', marker=m['Random'])


ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
plt.legend(handles=[BLEU4_random_plot, BLEU4_PAL_plot, BLEU4_PT4AL_plot, BLEU4_ALFA_plot,BLEU4_LL_plot,BLEU4_DOKT_plot], loc=4)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal'
}

x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(3.8, 20.2)
ax.set_ylim(74, 80.5)


#plt.plot([], [], ' ', label= r"Accuracy on 100% data = 93.50%" + '\n')
plt.legend(ncol=2, loc='lower right', prop=font1, frameon = True)
fig.tight_layout()
plt.xlabel('% of labeled data',font1, size = 15.5)
plt.ylabel('Mean Accuracy (%)',font1, size = 12)  
#plt.text(4.9, 15+19*0.95 , r"BLEU4 for 100% labeled data = 36.2 ", size = 14.5) 
#plt.title('CIFAR-10')
plt.grid(True)
fig.savefig('pretrain_mini.pdf',dpi=600,format='pdf')

