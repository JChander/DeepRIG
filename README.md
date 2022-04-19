# DeepRIG
A deep learning-based model for gene regulation networks (GRNs) inferrence from scRNA-seq data that transforms gene expression matrix into a correlation-based co-expression network and decouples the non-linear gene regulation patterns using relational graph convolution networks (GCNs).
 
See our manuscript for more details.
 
## Dependencies
 
**DeepRIG** is tested to work under Python 3.7. Other dependencies are list as follows:
 
* tensorflow 2.4
* numpy 1.19
* pandas 1.3
* h5py 2.10
* scipy 1.7
* scikit-learn 1.0
 
## Usage
### Inferring gene regulation networks with a trained model
Example: Inferring gene regulation networks from human embryonic stem cells (hESCs) dataset.  
```
>> python infering.py
```
 
### Train and test DeepRIG
Example: You can also train and test DeepRIG with a new dataset by following codes:
```
>> python main.py --input ./Datasets/mESC/  --network ./Datasets/mESC/refNetwork.csv  --output ./output/ 
```
### Outputs
* `Inferred_result_dataset_name.csv` Inferred gene regulation associations ranked by their edgeweights.
 
### Evaluation
Example: To evaluate the inferred results of DeepRIG from hESC dataset, run the following command:
```
python evaluate.py --pred_file ./output/ --network ./Datasets/hESC/refNetworks.csv
```
 
## Datasets
Demo datasets used in DeepRIG:
* hESC Human embryonic stem cells
* mESC mouse embryonic stem cells
* mDC mouse dendritic cells
* mHSC-E Erythroid lineages of mouse hematopoietic stem cells
* mHSC-L Lymphoid lineages of mouse hematopoietic stem cells
* mHSC-GM Granulocyte-macrophage lineages of mouse hematopoietic stem cells
