import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr('scGNNLTMG')
importr('reshape2')

def runLTMG(processedFile,ltmgFolder):
    robjects.globalenv['processedFile'] = processedFile
    robjects.globalenv['ltmgFolder'] = ltmgFolder

    robjects.r('''
        x <- read.csv(processedFile,header = T,row.names = 1,check.names = F)
        object <- CreateLTMGObject(x)
        object <-RunLTMG(object,Gene_use = 'all')
        exprs_proc_matrix <- object@OrdinalMatrix * object@InputData
        #WriteSparse(object,path=ltmgFolder,gene.name=FALSE, cell.name=FALSE)
        write.table(exprs_proc_matrix, file = ltmgFolder,row.names = T, quote = F,sep = ",")
        print('LTMG data saved!')
    ''')