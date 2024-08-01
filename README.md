# DeepRIG
A deep learning-based model for gene regulatary networks (GRNs) inferrence from scRNA-seq data that transforms gene expression matrix into a correlation-based co-expression network and decouples the non-linear gene regulation patterns using graph autoencoder model (GAE).
 
Please see our manuscript [here](https://doi.org/10.1371/journal.pgen.1010942) for more details.
 
## Dependencies
 
**DeepRIG** is tested to work under Python 3.7. Other dependencies are list as follows:
 
* tensorflow 2.4
* numpy 1.19
* pandas 1.3
* h5py 2.10
* scanpy 1.7
* scipy 1.7
* scikit-learn 1.0
 
## Installation
Installing within a conda environment is recommended. After Anaconda is installed in your OS, create a new environment.
```
>> conda create -n new_environ_name python=3.7
```
The `new_environ_name` is the new environment name with any name you prefer. Then activate your environment using following command:
```
>> conda activate new_environ_name
```
Installing all the dependencies recorded in the `requirements.txt` file in this repository using conda:
```
>> conda install --yes --file requirements.txt
```

## Usage
### Inferring gene regulatary networks from scRNA-seq data
To infer gene regulatary networks from scRNA-seq data using `main.py` script with the following options:  
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
python evaluate.py --pred_file ./output/Inferred_result_500_ChIP-seq_mESC.csv --network ./Datasets/500_ChIP-seq_mESC/500_ChIP-seq_mESC-networks.csv
```

### Cell-type specific GRNs inferring
**DeepRIG** also provides the cell-type specific GRNs inference. A Jupyter Notebook of the tutorial is accessible for the cell-type specific GRNs inference.
 
## Datasets
Demo datasets used in DeepRIG:
* hESC Human embryonic stem cells
* mESC mouse embryonic stem cells
* mDC mouse dendritic cells
* mHSC-E Erythroid lineages of mouse hematopoietic stem cells
* mHSC-L Lymphoid lineages of mouse hematopoietic stem cells
* mHSC-GM Granulocyte-macrophage lineages of mouse hematopoietic stem cells

## Citing
If you find our paper and code useful, please consider citing as follows:
```
@article{DeepRIG,
  title={Inferring gene regulatory network from single-cell transcriptomes with graph autoencoder model},
  author={Jiacheng, Wang and Yaojia, Chen and Quan, Zou},
  journal={PLoS Genetics},
  doi = {10.1371/journal.pgen.1010942},
  year={2023},
}
```
