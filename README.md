# PINN
Project for the deep learning course of Units. The aim of the project is to create a neural network using in the loss the residual of the PDEs such that the model learns how the model behaves other than what is represented by the data.
 
The problem we decided to work on is the 2-body problem of the Hamiltonian physics but there could be a major expansion toward the n-body problem.

A very important part of this project will be create a MLP such that the properties of the physics are still valid, like the fact that if we swap the input then the output should be exactly swapped (not an approximation, but exact values) and this property is also known as equivariance, another very important property is traslation.

## Aim of the project

We aim to create a PINN such that it evaluates all possible 2-body motions, not only one fixed problem. A very interesting prospective for this project is to try to find if the PINN trained only with 2-body problem (quite easy to evaluate with a simulator) is able to find a good solution with the 3-body problem or even with the n-body problem.

Another very interesting aspect is to see if the PINN for the 2-body problem is doing an approximation of a known solution for the 2-body problem (also known as the [action-angle variables](https://en.wikipedia.org/wiki/Action-angle_coordinates)).