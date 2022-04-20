# DeepRIG
A deep learning-based model for gene regulation networks (GRNs) inferrence from scRNA-seq data that transforms gene expression matrix into a correlation-based co-expression network and decouples the non-linear gene regulation patterns using relational graph convolution networks (GCNs).
 
See our manuscript for more details.
 
## Dependencies
 
**DeepRIG** is tested to work under Python 3.7. Other dependencies are list as follows:
 
* tensorflow 2.4
* numpy 1.19
* pandas 1.3
* h5py 2.10
* r-scgnnltmg 0.1
* scipy 1.7
* scikit-learn 1.0
 
## Usage
### Inferring gene regulation networks from scRNA-seq data
To infer gene regulation networks from scRNA-seq data using `main.py` script with the following options:  
* `input_path` string, the path of input dataset
* `output_path` string, the path of DeepRIG's output
* `cv` int, Folds for cross validation (Default 3)
* `ratio` int, Ratio of negative samples to positive samples (Default 1)
* `dim` int, The dimension of latent representations (Default 300)
* `hidden1` int, Number of unites in hidden layers (Default 200)
* `epochs` int, Number of epochs to train (Default 500)
* `learning_rate` float, Initial learning rate (Default 0.01)
* `dropout` float, Dropout rate in all layers in GCNs (Default 0.7)
 
Note: The names of gene expression file and ground truth file are expected as "DatasetName" + "-ExpressionData.csv"/"-network.csv". 
 
Example: Inferring GRNs from scRNA-seq of mouse embryonic stem cells (mESC) using DeepRIG by following codes:
```
>> python main.py --input_path ./Datasets/500_ChIP-seq_mESC/ --output_path ./output/ --cv 5
```
### Outputs
* `Inferred_result_dataset_name.csv` Inferred gene regulation associations ranked by their edgeweights.
 
### Evaluation
Example: To evaluate the inferred results of DeepRIG from mESC dataset, run the following command:
```
python evaluate.py --pred_file ./output/Inferred_result_500_ChIP-seq_mESC.csv --network ./Datasets/500_ChIP-seq_mESC/500_ChIP-seq-networks.csv
```
 
## Datasets
Demo datasets used in DeepRIG:
* hESC Human embryonic stem cells
* mESC mouse embryonic stem cells
* mDC mouse dendritic cells
* mHSC-E Erythroid lineages of mouse hematopoietic stem cells
* mHSC-L Lymphoid lineages of mouse hematopoietic stem cells
* mHSC-GM Granulocyte-macrophage lineages of mouse hematopoietic stem cells
