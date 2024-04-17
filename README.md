# Separator Fault Diagnosis
## Fault Diagnosis Study of Separator Equipment in Crude Oil Processing Systems
### Main Research Content
In offshore production platforms, horizontal three-phase separator is common process equipment. Its main function is to complete the dehydration and degassing of crude oil. The separator system is more complex, and its failure may cause significant economic losses and disastrous consequences. Therefore, it is critical to accurately and quickly identify where and why faults occur in the separator system. In this study, separator fault diagnosis model based on Bayesian networks is developed. Moreover, Sequential Monte Carlo simulation and physical model are introduced to overcome field problems such as missing separator failure data and inability of experts to provide accurate empirical knowledge. Using this model, 13 faults in a separator in an offshore crude oil processing system are successfully diagnosed. Meanwhile, proposed model is compared with deep neural network, convolutional neural network, and deep residual network, with accuracy rates of 100%, 91.34%, 87.99%, and 94.62%, respectively. Then, the diagnostic accuracy of each model for different faults is also compared in this paper under various signal-to-noise ratios. The results show that method proposed in this paper has better noise immunity compared to the other three models. Therefore, the accuracy and robustness of proposed model is further demonstrated. Finally, to analyze the fault-tolerance of proposed model, 2~3 error evidence is randomly entered. The results show that proposed model has better fault tolerance compared to data-driven Bayesian networks. 
### Tools Used for ModelS Building
#### Separator Simulation Model
The separator simulation model is built using Java language.
#### Bayesian Networks
The Bayesian Networks is built using Netcia.
#### CNN/DNN/RESNET
CNN, DNN and RESNET are built using Tensorflow.
#### Sequential Monte Carlo Simulation
Sequential Monte Carlo Simulation is built using Python.
#### Noisy-Max
The Noisy-Max is built using Python.
