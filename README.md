# PINN
Project for the deep learning course of Units. The aim of the project is to create a neural network using in the loss the residual of the PDEs such that the model learns how the model behaves other than what is represented by the data.
 
The problem we decided to work on is the 2-body problem of the Hamiltonian physics but there could be a major expansion toward the n-body problem.

A very important part of this project will be create a MLP such that the properties of the physics are still valid, like the fact that if we swap the input then the output should be exactly swapped (not an approximation, but exact values) and this property is also known as equivariance, another very important property is traslation.

## Aim of the project

We aim to create a PINN such that it evaluates all possible 2-body motions, not only one fixed problem. A very interesting prospective for this project is to try to find if the PINN trained only with 2-body problem (quite easy to evaluate with a simulator) is able to find a good solution with the 3-body problem or even with the n-body problem.

Another very interesting aspect is to see if the PINN for the 2-body problem is doing an approximation of a known solution for the 2-body problem (also known as the [action-angle variables](https://en.wikipedia.org/wiki/Action-angle_coordinates)).

## Possible other implementations/questions

- Start the training for the N body problem from the 2 body problem and see if beginning from that point improves the network
- if the mass of a single body is 0 in the 2 body problem, all of the accceleration should become (0,0). Does it happen? Should we create the network such that this is always assured?
- If the mass of a single body is 0 in the 3 body problem, it should behave exactly like the 2 body problem. Can we make it work? Should we start by multiplying the input by the mass?

## Results second question

If we consider the model of the 2 body problem and we proceed to set one of the 2 masses to 0, the body that doesn't have the null mass have effectly 0 acceleration and continue with a uniform rectilinear motion. The test shows the "trajectory of the null body", with an error of position, even if it doesn't exists but it isn't the important part.

# Results third question

If we consider the model of the N-body problem and we proceed to set one of the masses to 0, the other **2** bodies goes out like there is no acceleration, with a uniform rectilinear motion. This is very interesting, so the mass of a single body has so much influence over the other bodies, by completing eliminating it, likely a multiplication it erases everything else. I would like to avoid this, are needed more thoughts about this aspect.