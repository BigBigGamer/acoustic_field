import xlrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
import scipy.special as sp

plt.rc('text', usetex = True)
plt.rc('font', size=25, family = 'serif')
# plt.rc('text.latex',unicode=True)
# plt.rc('legend', fontsize=13)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')


def directPlate(theta, d):
    # theta - in degrees
    # d - in mm
    d /= 1000
    lmb = 1.5e-3     # 1.5mm
    tmp = (np.pi*d/lmb) * np.cos(theta/180*np.pi)
    # res = np.sin(tmp)/tmp
    res = 2 * sp.j1(tmp)/tmp
    return abs(res)

datafile = 'data/data.xlsx'

wb = xlrd.open_workbook(datafile)
sh = wb.sheet_by_index(0)

zero_angle = 65 

angle1 = np.array(sh.row_values(1)[1:]) - zero_angle
amp1 = np.array(sh.row_values(2)[1:])/456

angle2 = np.array(sh.row_values(4)[1:]) - zero_angle
amp2 = np.array(sh.row_values(5)[1:])/456

angle = angle1
angle = np.append(angle,angle2)
amp = amp1
amp = np.append(amp,amp2)

t_angles = np.linspace(-105,-75,1000)

plt.figure(figsize = (15,9))
# plt.polar(angle1/180*np.pi,amp1)
# plt.polar(angle2/180*np.pi,amp2)
# plt.xlim([-np.pi/10,np.pi/10])
# plt.polar((t_angles+90)/180*np.pi,directPlate(t_angles,18),'b-',label = 'Theory (d=18mm)')
plt.plot(angle,amp,'ko',label = 'Experiment',zorder = 3)
plt.plot(t_angles+90,directPlate(t_angles,10),'r-',label = 'Theory (d=10mm)')
plt.plot(t_angles+90,directPlate(t_angles,18),'b-',label = 'Theory (d=18mm)')
# plt.plot(-angle1,amp1,'ko',label = 'Experiment')
# plt.plot(-angle2,amp2,'ko')

plt.grid(True)
plt.legend()
plt.xlabel(r'Angle,$^{\circ}$')
# plt.ylabel('')
# plt.savefig('fig/direct.png',dpi=500)
plt.show()