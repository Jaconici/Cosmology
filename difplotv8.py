'''imports'''
from math import *
import matplotlib.pyplot as plt
from matplotlib import cm,colors,rc
import matplotlib.colors as colours
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import random
from matplotlib.ticker import LogLocator, AutoMinorLocator,MultipleLocator
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import glob
import os


def difplot(ylist,xlist,xlabel,ylabels,figx = 15,figy = 10,fontSize=40,DPI = 300,tickDirection='in',tickSize=1,font='serif',lineWidth=1.5,borderWidth = 3,color='random',cmap = None,yflip=False,xscale='linear',yscale='linear',name='dif.png',xscaled=1,yscaled=1,Mline=False,xspan=[],yspan=[],linestyle=None,path='',top = False,Loc='best',vertical = None,leg=['best','10'],xminor=0,yminor=0,text=None,numTicksy=50,numTicksx = 50,yTicks=[],xTicks=[],fill = []):
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
    plt.rcParams.update({
    "text.usetex": True,             # Enable LaTeX rendering
    "font.family": font,          # Use a serif font by default
    "text.latex.preamble": r"\usepackage{amsmath}",  # Optional, for advanced math formatting
})


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
    plt.rcParams['xtick.labelsize'] = fontSize
    plt.rcParams['ytick.labelsize'] = fontSize
    
    
    
    #tick direction
    plt.rcParams['xtick.direction'] = tickDirection
    plt.rcParams['ytick.direction'] = tickDirection
    
    
    
    #other options
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.unicode_minus'] = False
    if xscaled == 1:
        xscaled = [1 for i in range(0,len(xlist) +1)]
    if yscaled == 1:
        yscaled = [1 for i in range(0,len(ylist) + 1)]
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
    axs.set_xlabel(xlabel+xl,fontsize =  fontSize,labelpad = 25)
    axs.set_ylabel(ylabels[0],fontsize =  fontSize,labelpad = 25)
    if xspan !=[]:
        axs.set_xlim(xspan[0],xspan[1])
    if yspan !=[]:
        axs.set_ylim(yspan[0],yspan[1])  
    if yflip == True:
        axs.invert_yaxis()
    axs.set_xscale(xscale)
    axs.set_yscale(yscale)
    
    axs.minorticks_on()

    if np.size(ylabels) != 1:
        ylabels = [ylabels[i] for i in range(1,np.size(ylabels))]
    if leg != False:
        plt.legend(ylabels,loc=leg[0],fontsize=leg[1])
    if vertical != None:
        for v in vertical:
            plt.axvline(x=v[0], ymin=v[1], ymax=v[2],color=v[3],linestyle = v[4])
    if text != None:
        for t in text:
            plt.text(t[0],t[1],t[2],rotation=t[4],color = t[3],size=t[5])
    if xTicks !=[]:
        plt.xticks(xTicks)
    if yTicks != []:
        plt.yticks(yTicks)
    for tick in axs.get_xticklabels(minor = False):
        tick.set_y(-0.01)  # Adjust y-position of the tick labels

        
        
    for f in fill:
        if f[2] == None:
            # Create a polygon from the contour line
            polygon = Polygon(np.column_stack((f[0], f[1])), closed=True, edgecolor='none')
            
            # Use Path to create a mask outside the contour polygon
            fpath = Path(polygon.get_xy())
            outer_path = Path([
                [axs.get_xlim()[0], axs.get_ylim()[0]],
                [axs.get_xlim()[0], axs.get_ylim()[1]],
                [axs.get_xlim()[1], axs.get_ylim()[1]],
                [axs.get_xlim()[1], axs.get_ylim()[0]],
                [axs.get_xlim()[0], axs.get_ylim()[0]],
                    ])
            
            # Define a combined path that subtracts the contour path from the outer rectangle path
            combined_path = Path.make_compound_path(outer_path, fpath)
            
            # Add a patch for shading outside the contour region
            outside_patch = PathPatch(combined_path, facecolor=f[3], edgecolor='none', alpha=f[4])
            axs.add_patch(outside_patch)
        else:
            plt.fill_between(f[0], f[1], f[2], color=f[3], alpha=f[4])
    

    plt.savefig( path + name +  '.pdf',dpi=DPI, bbox_inches = "tight")
    
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
        for i in range (0,len(ylist[0])):
            linestyle[0].append("solid")
        for i in range (0,len(ylist[1])):
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
            axs[0].text(t[0],t[1],t[2],rotation=t[4],color = t[3],size=t[5])
        for t in text[1]:
            axs[0].text(t[0],t[1],t[2],rotation=t[4],color = t[3],size=t[5])
    plt.savefig(path + name + '.jpeg')
    

    
def contplot(xlist,ylist,zlist,xlabel,ylabel,zlabel,figx = 15,figy = 10,vmax=0,vmin= 0,fontSize=20,contours = None,zlist2 = [],alt = ['None'],tickDirection='in',tickSize=1,font='serif',lineWidth=1.5,borderWidth = 3,color='random',cmap='PuBu_r',xscale='linear',yscale='linear',name='dif.png',xspan=[],yspan=[],linestyle=None,path='Figures/',top = False,Loc='best',vertical = None,leg=['best','10'],returnPoints = False,text=None,logColors=True,lines=[],inLine = False,legend_boolean = None,algorithm = 'mpl2014',labelSize = 15):
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
    
    plt.rcParams['xtick.labelsize'] = labelSize  # Adjust as needed
    plt.rcParams['ytick.labelsize'] = labelSize  # Adjust as needed
    
    
    
    #other options
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

    

    
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'  # Axis label font weight
    plt.rcParams['mathtext.default'] = 'regular'  # Ensure math text is not bold


    
    if legend_boolean == None:
        legend_boolean = [True for i in contours[0][1]]
    
    legend_labels,legend_lines = [],[]
    fig, ax = plt.subplots(1, 1,figsize=[figx,figy])
    
    contour_points = []
    
    if logColors == True:
        if vmax !=0 and vmin!=0:
            Norm = colours.LogNorm(vmax=vmax,vmin=vmin)
        else:
            Norm = colours.LogNorm()
    elif logColors == False:
        if vmax !=0 and vmin!=0:
            Norm = colours.Normalize(vmax=vmax,vmin=vmin)
        else:
            Norm = colours.Normalize()
    for C in contours:
        cont = ax.contour(xlist,ylist,zlist,levels=[C[0]],linestyles = C[2],colors = C[1][0],linewidths = lineWidth,algorithm = algorithm)
        if inLine == True:
            ax.clabel(cont, inline=True, fontsize=leg[1],fmt = zlabel)
        elif legend_boolean[0] == True:
            legend_lines.append(Line2D([0], [0], color=C[1][0], linestyle=C[2], linewidth=lineWidth))
            legend_labels.append(zlabel)
            
        if returnPoints == True:
            for collection in cont.collections:
                for path in collection.get_paths():
                    contour_points.append(path.vertices)  # path.vertices is a 2D array of points (x, y)

    
        for j in range(0,len(zlist2)):
            cont2 = ax.contour(xlist,ylist,zlist2[j],levels=[C[0]],linestyles = C[j+3],colors = C[1][j+1],algorithm = algorithm)    
            if inLine == True:
                ax.clabel(cont2, inline=inLine, fontsize=leg[1],fmt = leg[2][j])
            elif legend_boolean[j + 1] == True:
                legend_lines.append(Line2D([0], [0], color=C[1][j + 1], linestyle=C[j+ 3], linewidth=lineWidth))
                legend_labels.append(leg[2][j])
            if returnPoints == True:
                for collection in cont2.collections:
                    for path in collection.get_paths():
                        contour_points.append(path.vertices)  # path.vertices is a 2D array of points (x, y)
        
    ax.legend(legend_lines, legend_labels,loc = leg[0],fontsize = leg[1])
        
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel,fontsize =  fontSize)
    ax.set_ylabel(ylabel,fontsize =  fontSize)

    
    ax.minorticks_on()



    if xspan !=[]:
        ax.set_xlim(xspan[0],xspan[1])
    if yspan !=[]:
        ax.set_ylim(yspan[0],yspan[1])   


    if text != None:
        for t in text:
            ax.text(t[0],t[1],t[2],rotation=t[4],color = t[3],size = t[5])
    for l in lines:
        ax.plot(l[0],l[1],color=l[2],linestyle=l[3])
    if alt != ['None']:
        for a in alt:
            ax.contour(a[0],a[1],a[2],levels=[a[3]],colors = a[4],linestyles = a[5],linewidths = a[6])
    if vertical != None:
        for v in vertical:
            plt.axvline(x=v[0], ymin=v[1], ymax=v[2],color=v[3],linestyle = v[4])
    
    
    if returnPoints == True:
        return contour_points
    else:
        plt.savefig(path + name + '.pdf')
    
        
    
    
def colplot(xlist,ylist,zlist,xlabel,ylabel,zlabel,figx = 15,figy = 10,vmax=0,vmin= 0,fontSize=20,contours = None,zlist2 = [],alt = ['None'],tickDirection='in',tickSize=1,font='serif',lineWidth=1.5,borderWidth = 3,color='random',cmap='PuBu_r',xscale='linear',yscale='linear',name='dif.png',xspan=[],yspan=[],linestyle=None,path='Figures/',top = False,Loc='best',vertical = None,leg=['best','10'],text=None,logColors=True,lines=[],inLine = False,legend_boolean = None,returnPoints = False):
    '''plots a subplot with same arguments as difplot except no option to scale, Mline, flip or cmap'''
    plt.rcParams.update({'font.size': fontSize,'font.family':font})
    plt.rcParams['axes.linewidth'] = borderWidth
    plt.rcParams.update({
    "text.usetex": True,             # Enable LaTeX rendering
    "font.family": "serif",          # Use a serif font by default
    "text.latex.preamble": r"\usepackage{amsmath}",  # Optional, for advanced math formatting
})

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
    
    plt.rcParams['xtick.labelsize'] = fontSize
    plt.rcParams['ytick.labelsize'] = fontSize
    
    
    
    #other options
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.unicode_minus'] = False
    contour_points = []
    
    if legend_boolean == None:
        legend_boolean = [True for i in contours[0][1]]
    
    legend_labels,legend_lines = [],[]
    fig, ax = plt.subplots(1, 1,figsize=[figx,figy])
    if logColors == True:
        if vmax !=0 and vmin!=0:
            Norm = colours.LogNorm(vmax=vmax,vmin=vmin)
        else:
            Norm = colours.LogNorm()
    elif logColors == False:
        if vmax !=0 and vmin!=0:
            Norm = colours.Normalize(vmax=vmax,vmin=vmin)
        else:
            Norm = colours.Normalize()
            
    pcm = ax.pcolor(xlist, ylist, zlist,cmap=cmap, shading='auto',norm = Norm)
    
    for C in contours:
        cont = ax.contour(xlist,ylist,zlist,levels=[C[0]],linestyles = C[2],colors = C[1][0],linewidths = lineWidth)
        if inLine == True:
            ax.clabel(cont, inline=True, fontsize=leg[1],fmt = zlabel)
        elif legend_boolean[0] == True:
            legend_lines.append(Line2D([0], [0], color=C[1][0], linestyle=C[2], linewidth=lineWidth))
            legend_labels.append(zlabel)
        if returnPoints == True:
            for collection in cont.collections:
                for path in collection.get_paths():
                    contour_points.append(path.vertices)
    
        for j in range(0,len(zlist2)):
            cont2 = ax.contour(xlist,ylist,zlist2[j],levels=[C[0]],linestyles = C[j+3],colors = C[1][j+1])    
            if inLine == True:
                ax.clabel(cont2, inline=inLine, fontsize=leg[1],fmt = leg[2][j])
            elif legend_boolean[j + 1] == True:
                legend_lines.append(Line2D([0], [0], color=C[1][j + 1], linestyle=C[j+ 3], linewidth=lineWidth))
                legend_labels.append(leg[2][j])
            if returnPoints == True:
                for collection in cont2.collections:
                    for path in collection.get_paths():
                        contour_points.append(path.vertices)
        
    ax.legend(legend_lines, legend_labels,loc = leg[0],fontsize = leg[1])
    ax.minorticks_on()   
 
    if xscale == 'log':
        ax.set_xscale(xscale)
        ax.xaxis.set_minor_locator(LogLocator())
    
    if yscale == 'log':
        ax.set_yscale(yscale)
        ax.xaxis.set_minor_locator(LogLocator())

        
        
    ax.set_xlabel(xlabel,fontsize =  fontSize)
    ax.set_ylabel(ylabel,fontsize =  fontSize)
    
    
    if xspan !=[]:
        ax.set_xlim(xspan[0],xspan[1])
    if yspan !=[]:
        ax.set_ylim(yspan[0],yspan[1])   


    if text != None:
        for t in text:
            ax.text(t[0],t[1],t[2],rotation=t[4],color = t[3],size = t[5])
    for l in lines:
        ax.plot(l[0],l[1],color=l[2],linestyle=l[3])
    if alt != ['None']:
        for a in alt:
            ax.contour(a[0],a[1],a[2],levels=[a[3]],colors = a[4],linestyles = a[5],linewidths = a[6])
    if vertical != None:
        for v in vertical:
            plt.axvline(x=v[0], ymin=v[1], ymax=v[2],color=v[3],linestyle = v[4])
    
    for tick in ax.get_xticklabels(minor = False):
        tick.set_y(-0.01)  # Adjust y-position of the tick labels

    if returnPoints == True:
        return contour_points
    else: 
        plt.savefig(path + name + '.pdf')
    
def read(path,master = [],output = 0,dtype = float,mi = 10000,length = 9,neg = 1, ref = [],size = None):
    Ys = []
    Mtemp,Mcheck = [],[]
    filetemp,iterable = '',[]
    sort = False 
    if master != []:
        for m in master:
            ms = format(m,f'.{1}e')
            ms = path + ms +'.txt'
            iterable.append([float(m),ms])
    else:
        for filename in glob.glob(path + "*"):
            name = os.path.basename(filename)[:length]
            iterable.append([float(name),filename])
        sort = True
    for it in iterable:
        Ystemp = []
        M,Mstring = it[0],format(it[0],f'.{1}e')
        if ref !=[]:
            Mcheck.append(float(Mstring))  
        data = np.genfromtxt(it[1], dtype=dtype)
        if output == 3:
            print('Reading file ' + Mstring)
        if size == None:
            size = len(data)
        for j in range(0,size):
            nan = False
            if np.size(data[j]) != 1:
                for k in range(0,np.size(data[j])):
                    if isnan(data[j][k]) == True:
                        if output != 1:
                            print('Nan detected in ' + Mstring)
                        nan = True
                if j <= mi and nan == False:
                    datareal = [data[j][k].real for k in range(0,np.size(data[j]))]
                    Ystemp.append(neg * datareal)
            else:
                if j <= mi and nan == False and isnan(data[j]) == False:
                    Ystemp.append(neg * data[j].real)
                else:
                    pass
        Mtemp.append(M)
        Ys.append(Ystemp)
    if ref !=[]:
        for m in ref:
            ms = format(m,f'.{1}e')
            if float(ms) in Mcheck:
                pass
            else:
                if output != 1:
                    print('Missing mass ' + str(m) + ' detected')
    if sort == True:    
        sorted_indices = sorted(range(len(Mtemp)), key=lambda i: Mtemp[i])
        Msorted = [Mtemp[i] for i in sorted_indices]
        Ysorted = [Ys[i] for i in sorted_indices]
        return [Msorted,Ysorted]
    else:
        return[Mtemp,Ys]
    
def read1D(path,master = [],output = 0,dtype = float,mi = 10000,correct = False,length = 9,neg = 1, ref = []):
    vals,fileVals = [],[]
    filetemp,iterable = '',[]
    
    for filename in glob.glob(path + "*"):
        name = os.path.basename(filename)[:length]
        iterable.append([float(name),filename])
    sort = True
    for it in iterable:
        valstemp = []
        fileVal,nameString = it[0],format(it[0],f'.{1}e')
        if output == 3:
            print('Reading file ' + nameString)
            
        data = np.genfromtxt(it[1], dtype=dtype)
        if np.size(data) != 1:
            nantag = False
            for i in range(0,np.size(data)):
                if dtype == np.complex64:
                    data[i] = float(data[i].real)
                if isnan(data[i]) == True:
                    if output != 1:
                        print('Nan detected in ' + Mstring)
                    nantag = True
            if nantag == False:
                vals.append(data)
                fileVals.append(fileVal)
        else:
            if dtype == np.complex64:
                data = float(data.real)
        
            if isnan(data) == True:
                if output != 1:
                    print('Nan detected in ' + Mstring)
                pass
            elif isnan(data) == False:
                vals.append(data)
                fileVals.append(fileVal)
    sorted_indices = sorted(range(len(fileVals)), key=lambda i: fileVals[i])
    valSorted = [vals[i] for i in sorted_indices]
    fileSorted = [fileVals[i] for i in sorted_indices]
    return [valSorted,fileSorted]
    
