'''imports'''
from math import *
import matplotlib.pyplot as plt
from matplotlib import cm,colors
import matplotlib.colors as colours
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import random
from matplotlib.ticker import LogLocator, AutoMinorLocator


def difplot(ylist,xlist,xlabel,ylabels,figx = 15,figy = 10,fontSize=40,DPI = 300,tickDirection='in',tickSize=1,font='serif',lineWidth=1.5,borderWidth = 3,color='random',cmap = None,yflip=False,xscale='linear',yscale='linear',name='dif.png',xscaled=1,yscaled=1,Mline=False,xspan=[],yspan=[],linestyle=None,path='Figures/',top = False,Loc='best',vertical = None,leg=['best','10'],xminor=0,yminor=0,text=None,numTicksy=50,numTicksx = 50):
    '''plots a solved differential equation'''
    '''======PARAMETERS======'''
    #ylist takes an array of arrays, each entry is a list of yvalues to be plotted. Same for xlist
    #ylables takes an array of at least 1 entry each which labels the functions on y axis. If more than one function plotted on y axis
    #... then the first entry to both is the common label, which appear as the axes labels, and the following entries (at least two) should differentiate
    #... the functions on the y axis
    #yscaled and xscaled are booleans which if set to True, scale the y and x axes by Mpl
    #xspan and yspan control the ylims and xlims, if left blank, set automatically
    #color takes either an array of colors, or the string 'random' which plots each line with a random color
    yl=''
    xl =''
    plt.rcParams.update({'font.size': fontSize,'font.family':font})
    plt.rcParams['axes.linewidth'] = borderWidth

    #x ticks dimension
    plt.rcParams['xtick.major.size'] = 12*tickSize
    plt.rcParams['xtick.major.width'] = 2*tickSize
    plt.rcParams['xtick.minor.size'] = 8*tickSize
    plt.rcParams['xtick.minor.width'] = 2*tickSize
    
    
    
    #y ticks dimension
    plt.rcParams['ytick.major.size'] = 12*tickSize
    plt.rcParams['ytick.major.width'] = 2*tickSize
    plt.rcParams['ytick.minor.size'] = 8*tickSize
    plt.rcParams['ytick.minor.width'] = 2*tickSize
    
    
    
    #tick direction
    plt.rcParams['xtick.direction'] = tickDirection
    plt.rcParams['ytick.direction'] = tickDirection
    
    
    
    #other options
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.unicode_minus'] = False
    if xscaled == 1:
        xscaled = [1 for i in range(0,np.size(xlist))]
    if yscaled == 1:
        yscaled = [1 for i in range(0,np.size(ylist))]
    if linestyle==None:
        linestyle=[]
        for i in range (0,np.size(ylist)+1):
            linestyle.append("solid")
    if color == 'random':
        if cmap==None:
            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(0 ,len(ylist))]
        if cmap != None:
            colmap  = cm.get_cmap(cmap)
            clist = [0+(i/len(ylist)) for i in range(0,len(ylist))]
            color = [colmap(i) for i in clist]
    for l in range(0 ,len(xlist)):
        xlist[l]= [i/xscaled[l] for i in xlist[l]]
    if xscaled !=1:
        xl = '' 
    for l in range(0 ,len(ylist)):
        ylist[l]= [i/yscaled[l] for i in ylist[l]]
    if yscaled !=1:
        yl = '' 
    fig = plt.figure(figsize=[figx,figy])
    axs = fig.subplots()
    if top != False:
        ax2 = axs.twiny()   
    if len(ylabels)==1:
        axs.plot(xlist[0],ylist[0],color=color[0],label=(ylabels[0]+' vs ' + xlabel),linestyle=linestyle[0],linewidth=lineWidth)
        if Mline == True:
            axs.plot([aT(M1)for i in range(0,10)],[ylist[0][int(np.size(ylist[0])/10)*i] for i in range(0,10)],linestyle="dashed",linewidth=lineWidth)
        if top != False:
            ax2.plot(xlist[0],ylist[0],color='None',linestyle=linestyle[0],linewidth=lineWidth)
    else:
        for l in range(0 ,len(ylist)):
            if l > len(ylabels)-2:
                axs.plot(xlist[l],ylist[l],color=color[l],linestyle=linestyle[l],linewidth=lineWidth)
            else:
                axs.plot(xlist[l],ylist[l],color=color[l],label=(ylabels[l+1]),linestyle=linestyle[l],linewidth=lineWidth)
            if top != False:
                ax2.plot(xlist[l],ylist[l],color='None',linewidth=lineWidth)
        if Mline == True:
            axs.plot([aT(M1)for i in range(0,10)],[ylist[0][int(np.size(ylist[0])/10)*i] for i in range(0,10)],linestyle="dashed")
    axs.set_xlabel(xlabel+xl,fontsize =  fontSize)
    axs.set_ylabel(ylabels[0]+yl,fontsize=fontSize)
    if xspan !=[]:
        axs.set_xlim(xspan[0],xspan[1])
    if yspan !=[]:
        axs.set_ylim(yspan[0],yspan[1])  
    if yflip == True:
        axs.invert_yaxis()
    axs.set_xscale(xscale)
    axs.set_yscale(yscale)
    if yscale == "log":
        locminy = LogLocator(base=10.0, subs=np.arange(2, 10,numTicksy) * .1,numticks=50)
        axs.yaxis.set_minor_locator(locminy)
    elif yscale == "linear":
        axs.yaxis.set_minor_locator(AutoMinorLocator())
    if xscale == "log":
        locminx = LogLocator(base=10.0, subs=np.arange(2, 10,numTicksx) * .1,numticks=50)
        axs.xaxis.set_minor_locator(locminx)
    elif yscale == "log":
        axs.xaxis.set_minor_locator(AutoMinorLocator())
    if top != False:
        ax2.set_xscale(xscale)
        ax2.set_xticklabels([str(round(float(top[0](ax2.get_xticks()[i])/M1),2)) for i in range(0,np.size(ax2.get_xticks()))])
        ax2.set_xlabel(top[1])
        #ax2.xaxis.set_major_formatter(FormatStrFormatter('{x:,.2f}'))
    #plt.tight_layout()
    if np.size(ylabels) != 1:
        ylabels = [ylabels[i] for i in range(1,np.size(ylabels))]
    if leg != False:
        plt.legend(ylabels,loc=leg[0],fontsize=leg[1])
    if vertical != None:
        for v in vertical:
            plt.axvline(x=v[0], ymin=v[1], ymax=v[2],color=v[3],linestyle = v[4])
    if text != None:
        for t in text:
            plt.text(t[0],t[1],t[2],rotation=t[4],color = t[3])
    plt.savefig(path + name + '.jpeg',dpi=DPI)
    
def difSubPlot(ylist,xlist,xlabel,ylabels,figx = 15,figy = 10,fontSize=40,tickDirection='in',tickSize=1,font='serif',lineWidth=1.5,borderWidth = 3,color='random',xscale='linear',yscale='linear',name='dif.png',xspan=[],yspan=[],linestyle=None,path='Figures/',top = False,Loc='best',vertical = None,leg=['best','10'],xminor=0,yminor=0,text=None,numTicksy=50,numTicksx = 50):
    '''plots a subplot with same arguments as difplot except no option to scale, Mline, flip or cmap'''
    plt.rcParams.update({'font.size': fontSize,'font.family':font})
    plt.rcParams['axes.linewidth'] = borderWidth

    #x ticks dimension
    plt.rcParams['xtick.major.size'] = 12*tickSize
    plt.rcParams['xtick.major.width'] = 2*tickSize
    plt.rcParams['xtick.minor.size'] = 8*tickSize
    plt.rcParams['xtick.minor.width'] = 2*tickSize
    
    
    
    #y ticks dimension
    plt.rcParams['ytick.major.size'] = 12*tickSize
    plt.rcParams['ytick.major.width'] = 2*tickSize
    plt.rcParams['ytick.minor.size'] = 8*tickSize
    plt.rcParams['ytick.minor.width'] = 2*tickSize
    
    
    
    #tick direction
    plt.rcParams['xtick.direction'] = tickDirection
    plt.rcParams['ytick.direction'] = tickDirection
    
    
    
    #other options
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.unicode_minus'] = False
    if linestyle==None:
        linestyle=[[],[]]
        for i in range (0,np.size(ylist[0])+1):
            linestyle[0].append("solid")
        for i in range (0,np.size(ylist[1])+1):
            linestyle[1].append("solid")
    if color == 'random':
        color = [[],[]]
        color[0] = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(0 ,len(ylist[0]))]
        color[1] = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(0 ,len(ylist[1]))]
    fig, axs = plt.subplots(2,figsize=[figx,figy])
    if top != False:
        ax2 = axs.twiny()  
    for i in range(0,len(ylabels)):
        if len(ylabels[i])==1:
            axs[i].plot(xlist[i][0],ylist[i][0],color=color[i][0],label=(ylabels[i][0]+' vs ' + xlabel),linestyle=linestyle[i][0],linewidth = lineWidth)
            if top != False:
                ax2.plot(xlist[i][0],ylist[i][0],color='None',linestyle=linestyle[i][0],linewidth = lineWidth)
        else:
            for l in range(0 ,len(ylist[i])):
                if l > len(ylabels[i])-2:
                    axs[i].plot(xlist[i][l],ylist[i][l],color=color[i][l],linestyle=linestyle[i][l],linewidth = lineWidth)
                else:
                    axs[i].plot(xlist[i][l],ylist[i][l],color=color[i][l],label=ylabels[i][l+1],linestyle=linestyle[i][l],linewidth = lineWidth)
                if top != False:
                    ax2.plot(xlist[i][l],ylist[i][l],color='None',linewidth = lineWidth)
    axs[0].set_xlabel(xlabel[0],fontsize =  fontSize)
    axs[0].set_ylabel(ylabels[0],fontsize=fontSize)
    axs[1].set_xlabel(xlabel[1],fontsize =  fontSize)
    axs[1].set_ylabel(ylabels[1],fontsize=fontSize)
    if xspan !=[]:
        axs[0].set_xlim(xspan[0][0],xspan[0][1])
        axs[1].set_xlim(xspan[1][0],xspan[1][1])
    if yspan !=[]:
        axs[0].set_ylim(yspan[0][0],yspan[0][1])  
        axs[1].set_ylim(yspan[1][0],yspan[1][1])  
    axs[0].set_xscale(xscale)
    axs[0].set_yscale(yscale)
    axs[1].set_xscale(xscale)
    axs[1].set_yscale(yscale)
    if yscale == "log":
        locminy = LogLocator(base=10.0, subs=np.arange(2, 10,numTicksy) * .1,numticks=50)
        axs[0].yaxis.set_minor_locator(locminy)
        axs[1].yaxis.set_minor_locator(locminy)
    elif yscale == "linear":
        axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        axs[1].yaxis.set_minor_locator(AutoMinorLocator())
    if xscale == "log":
        locminx = LogLocator(base=10.0, subs=np.arange(2, 10,numTicksx) * .1,numticks=50)
        axs[1].xaxis.set_minor_locator(locminx)
        axs[0].xaxis.set_minor_locator(locminx)
    elif yscale == "log":
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    if top != False:
        ax2.set_xscale(xscale)
        ax2.set_xticklabels([str(round(float(top[0](ax2.get_xticks()[i])/M1),2)) for i in range(0,np.size(ax2.get_xticks()))])
        ax2.set_xlabel(top[1])
        #ax2.xaxis.set_major_formatter(FormatStrFormatter('{x:,.2f}'))
    #plt.tight_layout()
    if np.size(ylabels) != 1:
        ylabels[0] = [ylabels[0][i] for i in range(1,np.size(ylabels[0]))]
        ylabels[1] = [ylabels[1][i] for i in range(1,np.size(ylabels[1]))]
    if leg != False:
        axs[0].legend(ylabels[0],loc=leg[0][0],fontsize=leg[0][1])
        axs[1].legend(ylabels[1],loc=leg[1][0],fontsize=leg[1][1])
    if vertical != None:
        for v in vertical[0]:
            axs[0].axvline(x=v[0], ymin=v[1], ymax=v[2],color=v[3],linestyle = v[4])
        for v in vertical[1]:
            axs[1].axvline(x=v[0], ymin=v[1], ymax=v[2],color=v[3],linestyle = v[4])
    if text != None:
        for t in text[0]:
            axs[0].text(t[0],t[1],t[2],rotation=t[4],color = t[3])
        for t in text[1]:
            axs[0].text(t[0],t[1],t[2],rotation=t[4],color = t[3])
    plt.savefig(path + name + '.jpeg')
    
def contplot(xlist,ylist,zlist,xlabel,ylabel,zlabel,figx = 15,figy = 10,fontSize=20,tickDirection='in',tickSize=1,font='serif',lineWidth=1.5,borderWidth = 3,color='random',cmap='PuBu_r',xscale='linear',yscale='linear',name='dif.png',xspan=[],yspan=[],linestyle=None,path='Figures/',top = False,Loc='best',vertical = None,leg=['best','10'],xminor=0,yminor=0,text=None,numTicksy=50,numTicksx = 50,logColors=True):
    '''plots a subplot with same arguments as difplot except no option to scale, Mline, flip or cmap'''
    plt.rcParams.update({'font.size': fontSize,'font.family':font})
    plt.rcParams['axes.linewidth'] = borderWidth

    #x ticks dimension
    plt.rcParams['xtick.major.size'] = 12*tickSize
    plt.rcParams['xtick.major.width'] = 2*tickSize
    plt.rcParams['xtick.minor.size'] = 8*tickSize
    plt.rcParams['xtick.minor.width'] = 2*tickSize
    
    
    
    #y ticks dimension
    plt.rcParams['ytick.major.size'] = 12*tickSize
    plt.rcParams['ytick.major.width'] = 2*tickSize
    plt.rcParams['ytick.minor.size'] = 8*tickSize
    plt.rcParams['ytick.minor.width'] = 2*tickSize
    
    
    
    #tick direction
    plt.rcParams['xtick.direction'] = tickDirection
    plt.rcParams['ytick.direction'] = tickDirection
    
    
    
    #other options
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.unicode_minus'] = False
    
    
    
    
    fig, ax = plt.subplots(1, 1,figsize=[figx,figy])
    if logColors == True:
        Norm = colours.LogNorm()
    elif logColors == False:
        Norm = Normalise()
    pcm = ax.pcolor(xlist, ylist, zlist,
                    norm=Norm,
                    cmap=cmap, shading='auto')
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    cb = fig.colorbar(pcm, ax=ax, extend='max')
    cb.set_label(zlabel,fontsize =  fontSize)
    ax.set_xlabel(xlabel,fontsize =  fontSize)
    ax.set_ylabel(ylabel,fontsize =  fontSize)

    if xspan !=[]:
        ax.set_xlim(xspan[0],xspan[1])
    if yspan !=[]:
        ax[0].set_ylim(yspan[0],yspan[1])   

    if yscale == "log":
        locminy = LogLocator(base=10.0, subs=np.arange(2, 10,numTicksy) * .1,numticks=50)
        ax.yaxis.set_minor_locator(locminy)
    elif yscale == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    if xscale == "log":
        locminx = LogLocator(base=10.0, subs=np.arange(2, 10,numTicksx) * .1,numticks=50)
        ax.xaxis.set_minor_locator(locminx)
    elif xscale == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    if text != None:
        for t in text:
            ax.text(t[0],t[1],t[2],rotation=t[4],color = t[3])

    plt.savefig(path + name + '.jpeg')