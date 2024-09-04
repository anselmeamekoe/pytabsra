# skorch tabsra

Python TabSRA package for TabSRA: An Attention based Self-Explainable Model for Tabular Learning

## Installation 
```pip install skorch-tabsra==0.0.1```
## Usage 
```python
from tabsra.skorch_tabsra import InputShapeSetterTabSRA,TabSRALinearRegressor

TabSRA = TabSRALinearRegressor(**params,callbacks=callbacks)
_ = TabSRA.fit(X_train_,Y_train_)
```
### Parameters & Methods 

**Parameters**

```module__dim_head``` : int, default=8. The attention head dimension , d_k in the paper. Typical values are {4,8,12}.


```module__n_hidden_encoder``` : int, default=2. The number of hidden layers in in the Key/Query encoder. Typical values are {1,2}.


```module__n_head``` : int, default=2. Number of SRA head/ensemble. Bigger value gives capacity to the model to produce less stable/robust explanations. Typical values are 1 or 2. 


```module__dropout_rate``` : float,default=0.0. The neuron dropout rate used in the Key/Query encorder during the training. 


```module__encoder_bias``` : bool. Whether to use bias term in the Key/Query encoder. 


```module__classifier_bias``` : bool, default=True. Whether to use bias term in the downstream linear classifier (regressor).


```module__attention_scaler``` : str, default='sqrt'. The scaling function for the attention weights.


```optimizer``` : default=torch.optim.Adam.


```lr``` : float, default=0.05. Learning rate used for the training. 


```batch_size``` : int, default=256.


```max_epochs``` :int, default=100. Maximal number of training iterations.


```iterator_train__shuffle``` : bool, default=True. Whether to shuffle the training batch iterator during the training


```verbose```: int, default=1. Set to 0 to  disable.


```random_state``` : int. default=42


```callbacks``` : The default is [InputShapeSetterTabSRA(regression=True)] which helps to infer the shape of the input data 


**Methods**

```fit(X,y,**fit_params)```: Fit the module


```predict(X)```: The prediction 


```predict_inference(X)```: Alternative inference method to predict. Suitable for very large datasets


```get_feature_attribution(X)```: Local feature attribution 


```get_attention(X)```: Produce attention weights


```get_weights()```: Returns the regression coefficients of the downstream linear part of the model

For remaining list of methods, please refer to https://skorch.readthedocs.io/en/stable/regressor.html

## Useful links
[Regression example](https://github.com/anselmeamekoe/pytabsra/blob/main/notebooks/Synthetic3_Regression_Example.ipynb)

[Bank Churn Classification example](https://github.com/anselmeamekoe/pytabsra/blob/main/notebooks/BankChurn_BinaryClassificationNet_ColumnTransfo_hidden1_paper.ipynb)
