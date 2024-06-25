import pandas as pd
import argparse
from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve
import numpy as np

def evaluateEPR(output, label, TFs, Genes):
	label_set = set(label['Gene1']+'|'+label['Gene2'])
	output= output.iloc[:len(label_set)]
	EPR = len(set(output['Gene1']+'|' +output['Gene2']) & label_set) / (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))
	return EPR

def evaluateAUPRratio(output, label, TFs, Genes):
	label_set_aupr = set(label['Gene1']+label['Gene2'])
	preds,labels,randoms = [] ,[],[]
	res_d = {}
	l = []
	p= []
	for item in (output.to_dict('records')):
			res_d[item['Gene1']+item['Gene2']] = item['EdgeWeight']
	for item in (set(label['Gene1'])):
			for item2 in  set(label['Gene1'])| set(label['Gene2']):
				if item+item2 in label_set_aupr:
					l.append(1)
				else:
					l.append(0)
				if item+ item2 in res_d:
					p.append(res_d[item+item2])
				else:
					p.append(-1)
	return average_precision_score(l,p), average_precision_score(l,p)/np.mean(l)

def evaluateAU(output, label):
	score = output.loc[:, ['EdgeWeight']].values
	label_dict = {}
	for row_index, row in label.iterrows():
		label_dict[row[0] + row[1]] = 1
	test_labels = []
	for row_index, row in output.iterrows():
		tmp = row[0]+str(row[1])
		if tmp in label_dict:
			test_labels.append(1)
		else:
			test_labels.append(0)
	test_labels = np.array(test_labels, dtype=bool).reshape([-1, 1])
	fpr, tpr, threshold = roc_curve(test_labels, score)
	auc_area = auc(fpr, tpr)
	# aucs.append(auc_area)
	precision, recall, _thresholds = precision_recall_curve(test_labels, score)
	aupr_area = auc(recall, precision)
	return auc_area, aupr_area

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Evaluate for the inferred results of DeepRIG from scRNA-seq data.')
	parser.add_argument('-p','--pred_file', type = str,
                        default = './output/Inferred_result_500_ChIP-seq_mESC.csv',
                        help='Path to inferred results file. Required. \n')
	parser.add_argument('-n','--network', type = str,
                        default = './Datasets/500_ChIP-seq_mESC/500_ChIP-seq_mESC-network.csv',
                        help='Path to network file to print network statistics. Optional. \n')
	args = parser.parse_args()

	groud_truth = args.network
	
	dataset  = groud_truth.split('/')[-2]
	output = pd.read_csv(args.pred_file, header = 0, sep=',')
	label = pd.read_csv(groud_truth, header = 0, sep = ',')

	output = output.groupby([output['Gene1'], output['Gene2']], as_index=False).mean()
	output = output[output['Gene1'] != output['Gene2']]
	auc, aupr = evaluateAU(output, label)
	print("========================Evaluation of Dataset: ", dataset, "========================")
	print("The AUC is:", '{:.4f}'.format(auc))
	print("The AUPR is:",'{:.4f}'.format(aupr))
	
	output['EdgeWeight'] = abs(output['EdgeWeight'])
	output = output.sort_values('EdgeWeight',ascending=False)
	
	TFs = set(label['Gene1'])
	Genes = set(label['Gene1'])| set(label['Gene2'])
	output = output[output['Gene1'].apply(lambda x: x in TFs)]
	output = output[output['Gene2'].apply(lambda x: x in Genes)]
	epr = evaluateEPR(output, label, TFs, Genes)
	aupr_1, aupr_ratio = evaluateAUPRratio(output, label, TFs, Genes)
	
	print("The EPR is:", '{:.4f}'.format(epr))
	print("The AUPR is:", '{:.4f}'.format(aupr_1))
	print("The AUPR ratio is:", '{:.4f}'.format(aupr_ratio))
