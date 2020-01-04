# Author: cym
import matplotlib.pyplot as plt
import numpy as np

# params={
#     'axes.labelsize': '35',
#     'xtick.labelsize':'27',
#     'ytick.labelsize':'27',
#     'lines.linewidth': 2 ,
#     'legend.fontsize': '27',
#     'figure.figsize' : '24, 9'
# }
# pylab.rcParams.update(params)

# [ca-hepth, gnutella04, grqc, PGP, ca-hepth time, gnutella04 time, grqc time, condmat time]
Y1 = [27.34,    172.32,     14.72,  18.76,  0.85, 6.79, 0.40, 1.51]  # MCF
Y2 = [137.46,   272.66,     69.38,  46.69,    4.51, 12.31, 1.91, 20.03]  # MIF
Y3 = [101.77,   270.84,     57.34,  23.15,    3.81,   11.75, 1.69, 17.76]  # degree_dis_effi
Y4 = [146.78,   211.04,     75.93,  49.17,     2.41,   12.56, 0.90, 9.13]  # MGF
Y5 = [147.98,   272.20,     76.19,  46.45,    3.99,   7.41, 1.85, 18.34]  # degree_dis__effi
Y6 = [148.01,   300.65,     76.74,  49.43,     1.94,   12.50, 0.80, 7.44]  # our
# Y7 = [50, 50, 50, 50, 51, 50, 50, 50] # Random

names = ["Hepth", "Gnutella04", "GRQC", "PGP", "Hepth", "Gnutella04", "GRQC", "PGP"]
col = 3
y1 = Y1[col]
y2 = Y2[col]
y3 = Y3[col]
y4 = Y4[col]
y5 = Y5[col]
y6 = Y6[col]
# y7 = Y7[col]

ind = np.arange(1)  # the x locations for the groups
width = 1
plt.bar(ind, y1, width, color='m', edgecolor="black", linewidth=1.5, label='MCF')
plt.bar(ind + width, y2, width, color='b', edgecolor="black", linewidth=1.5, label='MIF')  # ind+width adjusts the left start location of the bar.
plt.bar(ind + 2 * width, y3, width, color='r', edgecolor="black", linewidth=1.5, label='MIC')
plt.bar(ind + 3 * width, y4, width, color='y', edgecolor="black", linewidth=1.5, label='MGF')
plt.bar(ind + 4 * width, y5, width, color='c', edgecolor="black", linewidth=1.5, label='MGCRF')
plt.bar(ind + 5 * width, y6, width, color='g', edgecolor="black", linewidth=1.5, label='MREF')
# plt.bar(ind+6*width,y7, width,  color = 'k',    edgecolor="black",  linewidth=1.5,    label = 'Random')

# 图示如果和柱状图重合的话，调整一下
if col not in {5}:
    plt.bar(ind + 7 * width, [0], 2 * width, )
else:
    plt.bar(ind + 7 * width, [0], 4 * width, )
# plt.xticks(np.arange(5) + 2.5*width, ('10%','15%','20%','25%','30%'))


font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
plt.xlabel(names[col], font1)
if col < 4:
    plt.ylabel('Total Revenue', font1)
else:
    plt.ylabel('Running Time', font1)

# fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
# xticks = mtick.FormatStrFormatter(fmt)
# Set the formatter
axes = plt.gca()  # get current axes
# axes.yaxis.set_major_formatter(xticks) # set % format to ystick.
# axes.grid(True)
plt.xticks([])
# plt.legend(loc="upper right")
plt.legend(loc="best")


# 图表输出到本地
if col < 4:
    plt.savefig('./out/' + names[col] + '.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig('./out/' + names[col] + " time" '.png', dpi=300, bbox_inches='tight')
plt.show()
