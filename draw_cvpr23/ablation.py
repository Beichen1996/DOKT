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
m['Coreset']='>'
m['DOKT-T']='P'
m['DOKT-M']='^'
m['DOKT-T&M']='D'
m['MAE-Coreset']='X'






BLEU4_ratios = [item for item in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]

BLEU4_DOKT = [5.95,21.57,40.53,57.69,66.26,71.38,78.85,83.34,86.09,86.99  ]
BLEU4_DOKT_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_TM = [5.95,18.53,35.04,53.25,61.41,65.21,72.64,77.43,80.59,81.99 ]
BLEU4_TM_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0] 
BLEU4_DOKTT = [5.95,18.47,36.01,53.25,61.41,67.15,74.01,78.86,81.23,82.00 ]
BLEU4_DOKTT_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_DOKTM = [5.95,18.13,35.04,52.63,59.08,65.21,72.64,77.43,80.59,81.31  ]
BLEU4_DOKTM_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_MAECoreset = [5.95,19.84,39.41,56.62,63.00,69.19,76.02,80.13,83.07,83.95 ]
BLEU4_MAECoreset_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]
BLEU4_Coreset = [5.95,20.14,38.99,54.62,59.40,64.19,73.62,79.63,82.47,83.05  ]
BLEU4_Coreset_error = [0, 0, 0, 0, 0, 0, 0, 0,0,0]

BLEU4_ratios = BLEU4_ratios[1:];BLEU4_DOKT = BLEU4_DOKT[1:];BLEU4_TM = BLEU4_TM[1:];BLEU4_DOKTT = BLEU4_DOKTT[1:];BLEU4_DOKTM = BLEU4_DOKTM[1:];BLEU4_MAECoreset = BLEU4_MAECoreset[1:];BLEU4_Coreset = BLEU4_Coreset[1:]
fig, ax = plt.subplots(dpi=500)
plt_props()
#plt.rc('font',family='Times New Roman') 
BLEU4_DOKT_plot = plt.errorbar(BLEU4_ratios, BLEU4_DOKT, label='DOKT', marker=m['DOKT'] )

BLEU4_DOKTT_plot = plt.errorbar(BLEU4_ratios, BLEU4_DOKTT, label='DOKT-T', marker=m['DOKT-T'] )
BLEU4_DOKTM_plot = plt.errorbar(BLEU4_ratios, BLEU4_DOKTM, label='DOKT-M', marker=m['DOKT-M'])
BLEU4_TM_plot = plt.errorbar(BLEU4_ratios, BLEU4_TM, label='DOKT-T&M', marker=m['DOKT-T&M'] )
BLEU4_MAECoreset_plot = plt.errorbar(BLEU4_ratios, BLEU4_MAECoreset, label='MAE-Coreset', marker=m['MAE-Coreset'])
BLEU4_Coreset_plot = plt.errorbar(BLEU4_ratios, BLEU4_Coreset, label='Coreset', marker=m['Coreset'])


ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
plt.legend(handles=[BLEU4_Coreset_plot, BLEU4_DOKTT_plot, BLEU4_DOKTM_plot, BLEU4_MAECoreset_plot,BLEU4_TM_plot,BLEU4_DOKT_plot], loc=4)
  
font1 = {'family' : 'Times New Roman',
'weight' : 'normal'
}

x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(3.8, 20.2)
ax.set_ylim(17, 90)

#plt.text(4.2, 90, r"Accuracy on 100% data = 93.50%", fontsize=12, bbox=dict(facecolor='white', alpha=1, lw=0.8))

#plt.plot([], [], ' ', label= r"Accuracy on 100% data = 93.50%" + '\n')
plt.legend(ncol=2, loc='lower right', prop=font1, frameon = True)
fig.tight_layout()
plt.xlabel('% of labeled data',font1, size = 15.5)
plt.ylabel('Mean Accuracy (%)',font1, size = 12)  
#plt.text(4.9, 15+19*0.95 , r"BLEU4 for 100% labeled data = 36.2 ", size = 14.5) 
#plt.title('CIFAR-10')
plt.grid(True)
fig.savefig('ablation.pdf',dpi=600,format='pdf')

