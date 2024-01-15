# Separator Fault Diagnosis
## Fault Diagnosis Study of Separator Equipment in Crude Oil Processing Systems
### Main Research Content
In this study, a novel Bayesian fault diagnosis framework is proposed by combining the sequential Monte Carlo simulation with the physical model. Compared with the data model, the proposed framework can well explain the reasons for the faults and overcome the problem of missing data. Furthermore, this framework overcomes the shortcomings of traditional Bayesian networks in parameter acquisition and improves the efficiency of Bayesian network modeling. The main process of this framework is as follows: First, the structure of the Bayesian network is determined: system state layer, fault layer and fault symptom layer. Secondly, the parameters of the system state layer and the fault layer are determined using sequential Monte Carlo simulation, and the parameters of the fault symptom layer are determined using sequential Monte Carlo simulation and the physical model. Finally, the field fault information is input into the Bayesian networks for fault diagnosis. The proposed method has been successfully applied to fault diagnosis of a horizontal three-phase separator on an offshore platform. Meanwhile, we compare the proposed model with DNN, CNN, and RESNET models, and their accuracy rates are 100%, 90.64%, 86.66% and 94.54%, respectively. Therefore, it further proves the accuracy and effectiveness of the proposed model. In addition, to test the fault-tolerance of the proposed model, we randomly enter 2~3 erroneous evidence. The results show that the proposed model has better fault tolerance compared to data-driven Bayesian networks.
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
