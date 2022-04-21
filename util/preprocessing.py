##preprocessing single-cell expression data, filtering low-quality genes and cells, transforming data to the LTMG model, meanwhile sorting the cells by the pseudo-time 
import time
import argparse
import numpy as np
import pandas as pd
import pickle
import os.path
import scipy.sparse as sp
import scipy.io

parser = argparse.ArgumentParser(description='Arguments for Tools')
parser.add_argument('--expressionFile', type=str, default='Use_expression.csv',
                    help='expression File in csv')
parser.add_argument('--processedFile', type=str, default='processed_expression.csv',
                    help='processed expression File in csv')
parser.add_argument('--cellLabelFile', type = str, default = None,
                    help = 'Cell labels file in csv')
parser.add_argument('--LTMGDir', type=str, default=None,
                    help='directory of LTMG model produced.')
parser.add_argument('--orderedCellsFile', type=str, default=None,
                    help='Directory of odered cell expression matrix by pseudotime.')
parser.add_argument('--filetype', type=str, default='CSV',
                    help='select input filetype, 10X or CSV')
parser.add_argument('--delim', type=str, default='comma',
                    help='File delim type, comma or space: default(comma)')
parser.add_argument('--transform', type=str, default='log',
                    help='Whether transform')
parser.add_argument('--cellRatio', type=float, default=0.99,
                    help='cell ratio')
parser.add_argument('--geneRatio', type=float, default=0.99,
                    help='gene ratio')
parser.add_argument('--geneCriteria', type=str, default='variance',
                    help='gene Criteria')
parser.add_argument('--transpose', action='store_true', default=False,
                    help='whether transpose or not')
parser.add_argument('--isUMI', action='store_true', default=False,
                    help='whether UMI or not')


args = parser.parse_args()

def preprocessingCSV(expressionFile, processedFile, delim = 'comma', transform = 'log', cellRatio=0.9, geneRatio=0.9, geneCriteria = 'variance', transpose = False):
    '''
    preprocessing CSV files:
    transform='log' or None
    '''
    if not os.path.exists(expressionFile):
        print('Dataset ' + expressionFile + ' not exists!')

    print("Input scRNA data in CSV format, start to reading...")

    if delim == 'space':
        df = pd.read_csv(expressionFile, index_col = 0, delim_whitespace = True)
    elif delim == 'comma':
        df = pd.read_csv(expressionFile, index_col = 0)

    print('Expression File loading complete, start to filter low-quality genes and cells')

    if transpose ==True:
        df = df.T

    df1 = df[df.astype('bool').mean(axis=1) >= (1-geneRatio)]
    print('After preprocessing, {} genes remaining'.format(df1.shape[0]))
    criteriaGene = df1.astype('bool').mean(axis=0) >= (1-cellRatio)
    df2 = df1[df1.columns[criteriaGene]]
    print('After preprocessing, {} cells have {} nonzero'.format(
        df2.shape[1], geneRatio))

    if transform == 'log':
        df2 = df2.transform(lambda x: np.log(x + 1))
    print(df2.shape)
    df.to_csv(processedFile)

def preprocessingH5(expressionFile, sc_gene_list, processedFile, cellRatio=0.9, geneRatio=0.9, geneCriteria = 'variance', transpose = False):
    if not os.path.exists(expressionFile):
        print('Dataset ' + expressionFile + ' not exists!')
    
    print("Input scRNA data in H5 format, start to reading...")
    f = pd.HDFStore(expressionFile)
    df = f['RPKMs']
    gene_name_dict = get_gene_list(sc_gene_list)
    genes = df.columns
    gene_symbols = []
    for gene in genes:
        gene_symbols.append(gene_name_dict[str(gene)])

    df.columns = gene_symbols

    print('Expression File loading complete, start to filter low-quality genes and cells')

    if transpose ==True:
        df = df.T

    df1 = df[df.astype('bool').mean(axis=1) >= (1-geneRatio)]
    print('After preprocessing, {} genes remaining'.format(df1.shape[0]))
    criteriaGene = df1.astype('bool').mean(axis=0) >= (1-cellRatio)
    df2 = df1[df1.columns[criteriaGene]]
    print('After preprocessing, {} cells have {} nonzero'.format(
        df2.shape[1], geneRatio))

    if transform == 'log':
        df2 = df2.transform(lambda x: np.log(x + 1))
    print(df2.shape)
    df.to_csv(processedFile)

def get_gene_list(file_name):
    import re
    h={}
    s = open(file_name,'r') #gene symbol ID list of sc RNA-seq
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)',line)
        h[search_result.group(2)]=search_result.group(1).lower() # h [gene ID] = gene symbol
    s.close()
    return h

def transfOdiM2ExprsM(sparseMatrix, processedFile):
    df = pd.read_csv(sparseMatrix, header=None,
                     skiprows=1, delim_whitespace=True)
    counts = len(df) - 1
    proc_matrix = pd.read_csv(processedFile, header = 0, index_col = 0)
    for row in df.itertuples():
        # For the first row, it contains the number of genes and cells. Init the whole matrix
        if row[2] == counts:
            matrix = np.zeros((row[0], row[1]))
        else:
            matrix[row[2]-1][row[1]-1] = proc_matrix[row[2]-1][row[1]-1]

def computeCorr(expressionFile, corrMethod = 'pearson', threshold = 0.4):
    exprs = pd.read_csv(expressionFile, header = 0, index_col = 0, sep = "\t")
    exprsT = pd.DataFrame(exprs.T, columns = exprs.index, index = exprs.columns)
    corr = exprsT.corr(method= corrMethod)
    corr.values[corr.values < threshold] = 0.0
    return corr 

if __name__ == "__main__":
    start_time = time.time()

    # preprocessing
    print('Step1: Start filter and generating CSV')
    #if args.filetype == '10X':
        #expressionFilename = args.LTMGDir+args.datasetName+'/'+args.expressionFile
        # data = preprocessing10X(args.datasetDir, args.datasetName, args.LTMGDir+args.datasetName+'/'+args.expressionFile, args.transform, args.cellRatio, args.geneRatio, args.geneCriteria, args.geneSelectnum)
        #preprocessing10X(args.expressionFile, processedFile, args.transform,args.cellRatio, args.geneRatio, args.geneCriteria, args.geneSelectnum)
    if args.filetype == 'CSV':
        preprocessingCSV(args.expressionFile, args.processedFile, args.delim, args.transform,
                         args.cellRatio, args.geneRatio, args.geneCriteria, args.transpose)

    # start LTMG transforming
    from util.LTMG_Monocle_R import *
    print('Step2: Start infer LTMG and Pseudotime from Expression matrix')
    # run LTMG in R
    runLTMG(args.expressionFile, args.processedFile, args.cellLabelFile, args.LTMGDir, args.orderedCellsFile, args.isUMI)

    print("Preprocessing Done. Total Running Time: %s seconds" %
          (time.time() - start_time))
