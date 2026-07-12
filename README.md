# PINN
Project for the deep learning course of Units. The aim of the project is to create a neural network using in the loss the residual of the PDEs such that the model learns how the model behaves other than what is represented by the data.
 
The problem we decided to work on is the 2-body problem of the Hamiltonian physics but there could be a major expansion toward the n-body problem.

A very important part of this project will be create a MLP such that the properties of the physics are still valid, like the fact that if we swap the input then the output should be exactly swapped (not an approximation, but exact values) and this property is also known as equivariance, another very important property is traslation.
