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
    z0 = 0.0325
    lmb = 1.5e-3
    k = 2*np.pi/lmb
    R = np.sqrt(l**2 + (h - z0)**2)
    R1 = np.sqrt(l**2 + (h + z0)**2)
    deltaR = R - R1
    P = np.sqrt( 1/R**2 + 1/R1**2 - 2/(R*R1)*np.cos(k*deltaR) )
    return P

def approx_pressure(l, h):
    z0 = 0.033
    lmb = 1.5e-3
    R0 = np.sqrt( l**2 + h**2 )
    cos_theta1 = h/R0
    p = 2/R0*( np.sin( 2*np.pi/lmb*z0*cos_theta1 ) )
    return np.abs(p)

datafile = 'data/data.xlsx'

wb = xlrd.open_workbook(datafile)
sh = wb.sheet_by_index(2)

zero_height = 20.5

l70 = zero_height - parse_row(sh,1) 
amp70 = parse_row(sh,2)
amp70 /= np.amax(amp70)

l100 = zero_height - parse_row(sh,4)
amp100 = parse_row(sh,5)
amp100 /= np.amax(amp100)

l130 = zero_height - parse_row(sh,7)
amp130 = parse_row(sh,8)
amp130 /= np.amax(amp130)

h_th = np.linspace(1,19,1000)

# popt, pcov = sp.optimize.curve_fit(pressure, l1, amp1)
# print(popt)
l = 70
pr_th = pressure(l, h_th)
pr_th_appr = approx_pressure(l, h_th)
pr_th /= np.amax(pr_th)
pr_th_appr /= np.amax(pr_th_appr)

plt.figure(figsize = (17,8))
plt.plot(l70, amp70,'ko-', label = 'l = {} см'.format(l))
plt.plot(h_th, pr_th,'r-', label = 'Theory\nl = {} см,\n z0 = 3.25 см'.format(l))
# plt.plot(h_th, pr_th_appr,'b-', label = 'Theory2\nl = 130 см,\n z0 = 3.25см')


plt.grid(True)
plt.legend()
plt.xlabel('Растояние h, см')
plt.ylabel('Амплитуда')
# plt.savefig('fig/task31.png', dpi=500, bbox_inches = 'tight')
plt.show()