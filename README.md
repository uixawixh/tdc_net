# TDCNet(Two-Dimensional Crystal Neural Network)

## Version 1.0.1
1. XGBoost(regression and classification)
2. TDCNet(regression)
3. Feature generator(also support 3D crystal)

## usage
### 1. Install the dependent package
It is recommended to download the GPU version of torch. You can go to the https://pytorch.org/ to find the download method.
Install by pip or conda
```powershell
pip install torch scikit-learn pymatgen seaborn xgboost tqdm joblib
```
```powershell
conda install torch scikit-learn pymatgen seaborn xgboost tqdm joblib
```
Or install by requirement.txt(You should check your cuda version)
```powershell
pip install -r requirement.txt
```
### 2. Prepare the dataset
The data consists of two parts. One is the id_prop.csv file, which can contain more than two columns of data. We require that the first column is the prefix or full name of the crystal file, and the last column is the true label to be predicted. The middle column will be used as an additional single-column feature input.
<br>
example.csv:
```csv
crystal,feature1,feature2,...,label
Ti2C,1,10,...,1.0
Ti3C2,1,3,...,33.4
Ti3C2F2,2,2,...,-0.4
```
Place all listed crystal structures in the directory with the csv file.
### 3. Use it 
Use the help menu to get the available flags.
```powershell
python main.py -h
python predict.py -h
```
Then run it.
```powershell
python main.py examples/regression_example --model tdcnet
python main.py examples/classification_example --model xgboost
```
Replace the directory with your own directory to train your data.
## TODO
1.TDCNet classification
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       