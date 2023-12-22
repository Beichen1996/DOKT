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

BLEU4_DOKT = [34.13,40.34,42.58,44.33,46.29,47.96,49.31,50.42,51.75,52.56 ]
BLEU4_DOKT_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_LL = [34.13,38.85,40.82,42.70,44.79,46.67,47.77,49.11,50.85,51.39  ]
BLEU4_LL_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_ALFA = [34.13,39.60,42.09,43.85,45.73,47.43,48.69,49.87,51.23,52.03 ]
BLEU4_ALFA_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_PT4AL = [34.13,39.58,42.06,43.89,45.65,47.42,48.73,50.00,51.23,51.94 ]
BLEU4_PT4AL_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_PAL = [34.13,39.18,41.65,43.49,45.34,46.92,48.43,49.60,50.50,51.22 ]
BLEU4_PAL_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_random = [34.13,37.90,40.13,41.93,43.80,45.52,46.89,48.05,49.35,50.18 ]
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
ax.set_ylim(37.8, 53)



#plt.plot([], [], ' ', label= r"Accuracy on 100% data = 93.50%" + '\n')
plt.legend(ncol=2, loc='lower right', prop=font1, frameon = True)
fig.tight_layout()
plt.xlabel('% of labeled data',font1, size = 15.5)
plt.ylabel('Mean Accuracy (%)',font1, size = 12)  
#plt.text(4.9, 15+19*0.95 , r"BLEU4 for 100% labeled data = 36.2 ", size = 14.5) 
#plt.title('CIFAR-10')
plt.grid(True)
fig.savefig('pretrain_cifar100.pdf',dpi=600,format='pdf')

