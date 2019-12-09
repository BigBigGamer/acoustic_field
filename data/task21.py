import xlrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp

plt.rc('text', usetex = True)
plt.rc('font', size=25, family = 'serif')
# plt.rc('text.latex',unicode=True)
# plt.rc('legend', fontsize=13)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

def parse_row(sheet, row_num):
    tmp = sheet.row_values(row_num)[1:]
    lst = [s for s in tmp if s != '' ]
    return np.array(lst)

def pressure(l, h):
    # h - in cm
    # l - in cm
    h = h/100
    l = l/100
    z0 = 0.0335
    lmb = 1.5e-3
    k = 2*np.pi/lmb
    R = np.sqrt(l**2 + (h - z0)**2)
    R1 = np.sqrt(l**2 + (h + z0)**2)
    deltaR = R - R1
    P = np.sqrt( 1/R**2 + 1/R1**2 - 2/(R*R1)*np.cos(k*deltaR) )
    return P

def approx_pressure(l, h):
    z0 = 0.036
    lmb = 1.5e-3
    R0 = np.sqrt( l**2 + h**2 )
    cos_theta1 = h/R0
    p = 2/R0*( np.sin( 2*np.pi/lmb*z0*cos_theta1 ) )
    return np.abs(p)

datafile = 'data/data.xlsx'

wb = xlrd.open_workbook(datafile)
sh = wb.sheet_by_index(1)

zero_length = 18

l1 = parse_row(sh,1) + zero_length 
amp1 = parse_row(sh,2)/2
amp1 /= np.amax(amp1)

l2 = parse_row(sh,4) + zero_length 
amp2 = parse_row(sh,5)/2
amp2 /= np.amax(amp2)

l4 = parse_row(sh,7) + zero_length 
amp4 = parse_row(sh,8)/2
amp4 /= np.amax(amp4)

l_th = np.linspace(20,80,2000)
l_th = np.linspace(20,160,2000)
l_th_rare = np.linspace(20,80,100)

# popt, pcov = sp.optimize.curve_fit(pressure, l1, amp1)
# print(popt)
h = 4

pr_th = pressure(l_th, h)
pr_th_appr = approx_pressure(l_th_rare, h)
pr_th /= np.amax(pr_th)
pr_th_appr /= np.amax(pr_th_appr)

plt.figure(figsize = (17,8))
plt.plot(l4, amp4,'ko-', label = 'h = {} см'.format(h))
plt.plot(l_th, pr_th,'r-', label = 'Theory\nh = {} см,\n z0 = 3.35 см'.format(h))
# plt.plot(l_th_rare, pr_th_appr,'b.', label = 'Approximate theory\nh = 2.5 см,\n z0 = 3.6 см')


plt.grid(True)
plt.legend()
plt.xlabel('Растояние l, см')
plt.ylabel('Амплитуда')
plt.savefig('fig/task23.png', dpi=500, bbox_inches = 'tight')
plt.show()