#

# Author: Debashis Sahoo <dsahoo@ucsd.edu>
# License: GPLv2
# Date: April 9 2025

import pandas as pd
import numpy as np
import re
import requests
import json
import matplotlib
import mpl_toolkits.axes_grid1
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
import sklearn.metrics

try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

acolor = ["#00CC00", "#D8A03D","#EC008C",
    'cyan', "#B741DC", "#808285",
    'blue', 'black', 'green', 'red',
    'orange', 'brown', 'pink', 'purple']

def getPDF(cfile):
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(cfile)
    return pdf

def closePDF(pdf):
    import datetime
    d = pdf.infodict()
    d['Title'] = 'Plots'
    d['Author'] = 'Debashis Sahoo'
    d['Subject'] = "BoNE"
    d['Keywords'] = 'disease training validation ROC'
    d['CreationDate'] = datetime.datetime(2021, 10, 18)
    d['ModDate'] = datetime.datetime.today()
    pdf.close()
    return

def mergeRanks(group, start, exp, weight):
    X = np.array([[e[k-start] for e in exp] for k in group])
    arr = np.dot(X, np.array(weight))
    return arr

def getOrder(group, start, exp, weight):
    arr = mergeRanks(group, start, exp, weight)
    return [group[i] for i in np.argsort(arr)]

def getSName(name):
    l1 = re.split(": ", name)
    l2 = re.split(" /// ", l1[0])
    return l2[0]

def getRanksDf2(gene_groups, df_g, df_e, df_t):
    expr = []
    row_labels = []
    row_ids = []
    row_numhi = []
    ranks = []
    g_ind = 0
    counts = []
    noisemargins = []
    for k in range(len(df_e)):
        count = 0
        order = range(2, df_e[k].shape[1])
        avgrank = [0 for i in order]
        noisemargin = 0
        for j in df_g[k]['idx']:
            e = df_e[k].iloc[j,:].to_list()
            t = df_t[k]['thr2'][j]
            if e[-1] == "":
                continue
            v = np.array([float(e[i]) if e[i] != "" else 0 for i in order])
            te = []
            sd = np.std(v)
            for i in order:
                if (e[i] != ""):
                    v1 = (float(e[i]) - t) / 3;
                    if sd > 0:
                        v1 = v1 / sd
                else:
                    v1 = -t/3/sd
                avgrank[i-2] += v1
                te.append(v1)
            nv1 = 0.5/3
            if sd > 0:
                nv1 = nv1 / sd
            noisemargin += nv1 * nv1
            expr.append(te)
            nm = getSName(e[1])
            row_labels.append(nm)
            row_ids.append(e[0])
            v1 = [g_ind, sum(v > t)]
            if g_ind > 3:
                v1 = [g_ind, sum(v <= t)]
            else:
                v1 = [g_ind, sum(v > t)]
            row_numhi.append(v1)
            count += 1
            #if count > 200:
            #    break
        ranks.append(avgrank)
        noisemargins.append(noisemargin)
        g_ind += 1
        counts += [count]
    print(counts)
    return ranks, noisemargins, row_labels, row_ids, row_numhi, expr

def cAllPvals(lval, atypes):
    for i in range(len(lval)):
        for j in range(i +1, len(lval)):
            if len(lval[i]) <= 0:
                continue
            if len(lval[j]) <= 0:
                continue
            #print(lval[i])
            #print(lval[j])
            t, p = scipy.stats.ttest_ind(lval[i],lval[j], equal_var=False)
            desc = "%s vs %s %.3g, %.3g" % (atypes[i], atypes[j], t, p)
            print(desc)
    return

def adj_light(color, l=1, s=1):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, l * c[1])),
                 max(0, min(1, s * c[2])))

def barTop(tax, atypes, color_sch1, params):
    spaceAnn = 70
    widthAnn = 3
    tAnn = 1
    if 'spaceAnn' in params:
        spaceAnn = params['spaceAnn']
    if 'widthAnn' in params:
        widthAnn = params['widthAnn']
    if 'tAnn' in params:
        tAnn = params['tAnn']
    for i in range(len(atypes)):
        tax.add_patch(matplotlib.patches.Rectangle( (i *spaceAnn, 0), widthAnn, 3,
                                        facecolor=color_sch1[i], edgecolor="none", alpha=1.0))
        tax.text(i * spaceAnn + widthAnn + tAnn, 1, atypes[i], rotation='horizontal',
                 ha='left', va='center', fontsize=12)
    return

def plotTitleBar(cval, atypes, params):
    dpi = 100
    if 'dpi' in params:
        dpi = params['dpi']
    w,h = (5, 0.8)
    if 'w' in params:
        w = params['w']
    if 'h' in params:
        h = params['h']
    color_sch1 = ["#3B449C", "#B2509E","#EA4824"]
    color_sch1 = ["#00CC00", "#EFF51A","#EC008C", "#F7941D", "#808285",
            'cyan', 'blue', 'black', 'green', 'red']
    if 'acolor' in params:
        color_sch1 = params['acolor']
    if 'cval' in params:
        cval = params['cval']

    ax = None
    if 'ax' in params:
        ax = params['ax']
    if ax is None:
        fig = plt.figure(figsize=(w,h), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
    nAt = len(cval[0])
    extent = [0, nAt, 0, 5]
    ax.axis(extent)
    cmap = matplotlib.colors.ListedColormap(color_sch1)
    boundaries = range(len(color_sch1) + 1)
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    #ax.imshow(cval, interpolation='none', cmap=cmap, \
    #                  norm=norm, extent=extent, aspect="auto")
    y = [0, 5]
    x = np.arange(nAt + 1)
    ax.pcolormesh(x, y, cval, cmap=cmap, norm=norm, zorder=-1.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(top=False, left=False, bottom=False, right=False)
    ax.set_xticks(np.arange(0, nAt, 1))
    ax.grid(which='major', alpha=0.2, linestyle='-', linewidth=0.5,
            color='black', zorder=2.0)
    for edge, spine in ax.spines.items():
                spine.set_visible(False)

    divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax)
    width = mpl_toolkits.axes_grid1.axes_size.AxesX(ax, aspect=1./20)
    spaceAnn = 70
    widthAnn = 3
    tAnn = 1
    if 'spaceAnn' in params:
        spaceAnn = params['spaceAnn']
    if 'widthAnn' in params:
        widthAnn = params['widthAnn']
    if 'tAnn' in params:
        tAnn = params['tAnn']
    pad = mpl_toolkits.axes_grid1.axes_size.Fraction(0.1, width)
    lax = divider.append_axes("top", size="100%", pad="20%", frame_on=False)
    lax.axison = False
    lax.axis(extent)
    lax.set_xticklabels([])
    lax.set_yticklabels([])
    lax.grid(False)
    lax.tick_params(top=False, left=False, bottom=False, right=False)
    if 'atypes' in params:
        atypes = params['atypes']
    barTop(lax, atypes, color_sch1, params)
    return ax

def plotViolin(data, atypes, params):
    df = pd.DataFrame()
    df['score'] = [k for i in range(len(data)) for k in data[i]]
    df['category'] = [atypes[i] for i in range(len(data)) for k in data[i]]
    m1 = []
    pvals = []
    for i in range(1, len(data)):
        if len(data[i]) <= 0:
            m1 += [0]
            pvals += [""]
            continue
        m1 += [max(data[i]) + (max(data[i]) - min(data[i])) * 0.1]
        t, p = scipy.stats.ttest_ind(data[0],data[i], equal_var=False)
        if (p < 0.05):
            pvals += ["p=%.3g" % p]
        else:
            pvals += [""]
    dpi = 100
    if 'dpi' in params:
        dpi = params['dpi']
    w,h = (1.5 * len(atypes), 4)
    if 'w' in params:
        w = params['w']
    if 'h' in params:
        h = params['h']
    color_sch1 = acolor
    if 'acolor' in params:
        color_sch1 = params['acolor']
    sns.set()
    sns.set_style("white")
    sns.set_style({'text.color': '.5',
        'xtick.color':'.5', 'ytick.color':'.5', 'axes.labelcolor': '.5'})
    sns.set_context("notebook")
    sns.set_palette([adj_light(c, 1.5, 1) for c in color_sch1])
    ax = None
    ax = None
    if 'ax' in params:
        ax = params['ax']
    if ax is None:
        fig,ax = plt.subplots(figsize=(w,h), dpi=dpi)
    width = 1
    height = 1
    if 'width' in params:
        width = params['width']
    if 'vert' in params and params['vert'] == 1:
        ax = sns.violinplot(x="category", y="score", inner='quartile',
                hue='category', linewidth=0.5, width=width, ax = ax, data=df,
                order = atypes)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax = sns.swarmplot(x="category", y="score", palette = 'dark:blue', alpha=0.2,
                hue='category', ax=ax, data=df, order = atypes)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("")
        pos = range(len(atypes))
        for tick,label in zip(pos[1:],ax.get_xticklabels()[1:]):
            ax.text(pos[tick], m1[tick - 1], pvals[tick - 1],
                    horizontalalignment='center', size=12,
                    color='0.3')
        ax.yaxis.grid(True, clip_on=False)
    else:
        ax = sns.violinplot(x="score", y="category", inner='quartile',
                hue='category', linewidth=0.5, width=width, ax = ax, data=df,
                order = atypes)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax = sns.swarmplot(x="score", y="category", palette = 'dark:blue', alpha=0.2,
                hue='category', ax=ax, data=df, order = atypes)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel("")
        pos = range(len(atypes))
        for tick,label in zip(pos[1:],ax.get_yticklabels()[1:]):
            ax.text(m1[tick - 1], pos[tick]-0.5, pvals[tick - 1],
                    horizontalalignment='center', size=12,
                    color='0.3')
        ax.xaxis.grid(True, clip_on=False)
    return ax

def plotViolinBar(ana, desc=None):
    fig = plt.figure(figsize=(4,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=3)
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': [], 'ax': ax1, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    res = ana.getROCAUC()
    ax.text(len(ana.cval[0]), 4, res)
    if desc is not None:
        ax.text(-1, 2, desc, horizontalalignment='right',
                    verticalalignment='center')
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
            'genes': [], 'ax': ax2, 'acolor': acolor, 'vert': 0}
    ax = ana.printViolin(None, params)
    return fig


urlbase="http://hegemon.ucsd.edu/Tools/explore.php"
class HegemonUtil:
    @staticmethod
    def uniq(mylist):
      used = set()
      unique = [x for x in mylist if x not in used and (used.add(x) or True)]
      return unique
    @staticmethod
    def getHegemonDataset(dbid, urlbase=urlbase):
        url = urlbase + "?go=getdatasetjson&id=" + dbid
        response = requests.get(url)
        obj = json.loads(response.text)
        return  obj
    @staticmethod
    def getHegemonPatientInfo(dbid, urlbase=urlbase):
        url = urlbase + "?go=getpatientinfojson" + "&id=" + dbid
        response = requests.get(url)
        obj = json.loads(response.text)
        return  obj
    @staticmethod
    def getHegemonPatientData(dbid, name, urlbase=urlbase):
        hdr = HegemonUtil.getHegemonPatientInfo(dbid, urlbase)
        clinical = 0
        if name in hdr:
            clinical = hdr.index(name)
        url = urlbase + "?go=getpatientdatajson" + \
            "&id=" + dbid + "&clinical=" + str(clinical)
        response = requests.get(url)
        obj = json.loads(response.text)
        return  obj
    @staticmethod
    def getHegemonDataFrame(dbid, genelist=None, pGroups=None, urlbase=urlbase):
        genes =''
        if genelist is not None:
            genes = ' '.join(genelist)
        groups = ''
        if pGroups is not None:
            for i in range(len(pGroups)):
                str1 = "=".join([str(i), pGroups[i][0], ":".join(pGroups[i][2])])
                if i == 0:
                    groups += str1
                else:
                    groups = groups + ';' + str1
        url = urlbase
        opt = {'go': 'dataDownload', 'id': dbid, 'genes': genes, 'groups' : groups}
        response = requests.post(url, opt)
        data = StringIO(response.text)
        df = pd.read_csv(data, sep="\t")
        return df
    @staticmethod
    def getHegemonThrFrame(dbid, genelist=None, urlbase=urlbase):
        genes =''
        if genelist is not None:
            genes = ' '.join(genelist)
        url = urlbase
        opt = {'go': 'dataDownload', 'id': dbid, 'genes': genes, 'groups' : '',
              'param': 'type:thr'}
        response = requests.post(url, opt)
        data = StringIO(response.text)
        df = pd.read_csv(data, sep="\t", header=None)
        df.columns=['ProbeID', 'thr1', 'stat', 'thr0', 'thr2']
        return df
    @staticmethod
    def getHegemonGeneIDs(dbid, genelist=None, urlbase=urlbase):
        genes =''
        if genelist is not None:
              genes = ' '.join(genelist)
        url = urlbase
        opt = {'go': 'dataDownload', 'id': dbid, 'genes': genes, 'groups' : '',
              'param': 'type:geneids'}
        response = requests.post(url, opt)
        data = StringIO(response.text)
        df = pd.read_csv(data, sep="\t")
        return df
    @staticmethod
    def getHegemonPlots(dbid, gA, gB, urlbase=urlbase):
      url = urlbase + "?go=getplotsjson&id=" + dbid + \
              "&A=" + str(gA) + "&B=" + str(gB)
      response = requests.get(url)
      obj = json.loads(response.text)
      return  obj
    @staticmethod
    def getHegemonData(dbid, gA, gB, urlbase=urlbase):
      url = urlbase + "?go=getdatajson&id=" + dbid + \
              "&A=" + str(gA) + "&B=" + str(gB)
      response = requests.get(url)
      obj = json.loads(response.text)
      return  obj

hu = HegemonUtil
class BooleanAnalysis:
    def __init__(self, urlbase=urlbase):
        self.state = []
        self.params = {}
        self.start = 2
        self.end = 2
        self.urlbase = urlbase
        return

    def aRange(self):
        return range(self.start, self.end + 1)

    def getTitle(self):
        title = self.name + " (" + self.source + "; n = " + str(self.num) + ")"
        return title

    def printInfo(self):
        print(self.name + " (n = " + str(self.num) + ")")
        print(self.source + " " + self.dbid)
        print(len(self.order), [len(i) for i in self.state], \
              self.source, self.dbid)
        return
    
    def prepareDataDf(self, dbid, urlbase=urlbase):
        self.dbid = dbid
        self.dataset = hu.getHegemonDataset(self.dbid, urlbase)
        self.num = self.dataset[2]
        self.name = self.dataset[1]
        self.source = self.dataset[3]
        obj = hu.getHegemonPatientData(self.dbid, 'time', urlbase)
        self.headers = obj[0]
        self.hhash = {}
        self.start = 2;
        self.end = len(self.headers) - 1
        for i in range(len(self.headers)):
            self.hhash[self.headers[i]] = i
        return
    
    def initData(self, atype, atypes, ahash):
        for i in range(len(atypes)):
            ahash[atypes[i]] = i
        aval = [ahash[i] if i in ahash else None for i in atype]
        expg = [i for i in self.aRange() if aval[i] is not None]
        self.state = [[i for i in range(len(atype)) if aval[i] == k]
                for k in range(len(atypes))]
        self.aval = aval
        self.atype = atype
        self.atypes = atypes
        self.order = expg
        self.printInfo()
        return

    def getSurvName(self, name):
        return hu.getHegemonPatientData(self.dbid, name, self.urlbase)[1]

    def getExprData(self, name):
        return hu.getHegemonData(self.dbid, name, "", self.urlbase)[1]

    def getThrData(self, name):
        obj = hu.getHegemonThrFrame(self.dbid, [name])
        thr = [obj['thr1'][0], obj['stat'][0], obj['thr0'][0], obj['thr2'][0]]
        return thr

    def getArraysThr (self, id1, thr = None, type1 = None):
        res = []
        expr = self.getExprData(id1);
        thr_step = self.getThrData(id1);
        thr = hu.getThrCode(thr_step, thr_step[0], thr);
        for i in self.aRange():
          if (thr is None):
             res.append(i)
          elif (expr[i] == ""):
             continue
          elif (type1 == "hi" and float(expr[i]) >= thr):
             res.append(i)
          elif (type1 == "lo" and float(expr[i]) < thr):
             res.append(i)
          elif (type1 is not None and type1 != "lo" and type1 != "hi" \
                  and float(expr[i]) >= thr and float(expr[i]) <= float(type1)):

             res.append(i)
        return res

    def getArraysAll (self, *data):
        res = self.aRange()
        for i in range(0, len(data), 3):
          r = self.getArraysThr(data[i], data[i+1], data[i+2])
          res = list(set(res) & set(r))
        return res;

    def orderDataDf(self, gene_groups, weight):
        data_g = []
        data_e = []
        data_t = []
        for k in gene_groups:
            df_g = hu.getHegemonGeneIDs(self.dbid, k, self.urlbase)
            df_e = hu.getHegemonDataFrame(self.dbid, k, None, self.urlbase)
            df_t = hu.getHegemonThrFrame(self.dbid, k, self.urlbase)
            df_e.fillna(0,inplace=True)
            rhash = {}
            for i in range(df_t.shape[0]):
                rhash[df_t.iloc[i,0]] = i
            order = [rhash[df_e.iloc[i,0]] for i in range(df_e.shape[0])]
            df_t = df_t.reindex(order)
            df_t.reset_index(inplace=True)
            rhash = {}
            for i in df_e.index:
                rhash[df_e.iloc[i,0]] = i
            df_g['idx'] = [rhash[df_g.iloc[i,0]] if df_g.iloc[i,0] in rhash
                    else None for i in df_g.index]
            df_g = df_g.dropna()
            df_g['idx'] = df_g['idx'].astype(np.int64)
            data_g.append(df_g)
            data_e.append(df_e)
            data_t.append(df_t)
        self.col_labels = self.headers[self.start:]
        if len(gene_groups) > 0:
            self.col_labels = data_e[0].columns[self.start:]
        self.chash = {}
        for i in range(len(self.col_labels)):
            self.chash[self.col_labels[i]] = i
        compositres = getRanksDf2(gene_groups, data_g, data_e, data_t)
        ranks, noisemargins, row_labels, row_ids, row_numhi, expr = compositres
        i1 = getOrder(self.order, self.start, ranks, weight)
        index = np.array([i - self.start for i in i1])
        self.cval = np.array([[self.aval[i] for i in i1]])
        #self.data = np.array(expr)[:,index]
        self.ind_r = np.array(sorted(range(len(row_labels)),
            key=lambda x: (row_numhi[x][0], row_numhi[x][1])))
        row_labels = [row_labels[i] for i in self.ind_r]
        row_ids = [row_ids[i] for i in self.ind_r]
        self.data = np.array([expr[i] for i in self.ind_r])[:,index]
        self.f_ranks = mergeRanks(range(self.start, len(self.headers)),
                self.start, ranks, weight)
        f_nm = 0
        for i in range(len(gene_groups)):
            f_nm += abs(weight[i]) * noisemargins[i]
        self.noisemargin = 0.5/3
        if f_nm > 0:
            self.noisemargin = np.sqrt(f_nm)
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.index = index
        return

    def getScores(ana, ahash = None):
        lval = [[] for i in ana.atypes]
        cval = ana.cval[0]
        if ahash is None:
            score = [ana.f_ranks[i - ana.start] for i in ana.i1]
        else:
            score = [ana.f_ranks[ana.i1[i] - ana.start] \
                    for i in range(len(ana.i1)) \
                    if ana.cval[0][i] in  ahash ]
            cval = [ana.cval[0][i] \
                    for i in range(len(ana.i1)) \
                    if ana.cval[0][i] in  ahash ]
        for i in range(len(cval)):
            lval[cval[i]] += [score[i]]
        return lval, score

    def printAllPvals(self, ahash = None, params = None):
        lval, score = self.getScores(ahash=ahash)
        cAllPvals(lval, self.atypes)
        return
    
    def getROCAUCspecific(ana, m=0, n=1):
        actual = [ana.aval[i] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        score = [ana.f_ranks[i - ana.start] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        if (len(actual) == 0):
            return "0.5"
        fpr, tpr, thrs = sklearn.metrics.roc_curve(actual, score, pos_label=n)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        return "%.2f" % roc_auc

    def getROCAUC(ana):
        res = []
        for k in range(1, len(ana.atypes)):
            v = ana.getROCAUCspecific(0, k)
            res += [v]
        return ",".join(res)

    def printTitleBar(self, params):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax = plotTitleBar(self.params['cval'], \
                self.params['atypes'], self.params)
        return ax

    def printViolin(self, ahash = None, params = None):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        lval, score = self.getScores(ahash=ahash)
        atypes = [str(self.params['atypes'][i]) + "("+str(len(lval[i]))+")"
                for i in range(len(self.params['atypes']))]
        ax = plotViolin(lval, atypes, self.params)
        return ax

def getMSigDB(gs):
    url = "https://www.gsea-msigdb.org/gsea/msigdb/download_geneset.jsp?geneSetName=" + gs + "&fileType=txt"
    df = pd.read_csv(url, sep="\t")
    df.columns.values[0] = 'ID'
    l1 = [list(df.ID[1:])]
    wt1 = [1]
    return wt1, l1
BooleanAnalysis.getMSigDB = getMSigDB
