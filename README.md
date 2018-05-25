# VecOptim

There is a certain domain of tasks, that require you to change some characteristics of an object, without changing it's inner
structure. 

For example if you want to change a shape of a face in the photo without changing nose, mouth or anything else on it, or if you
want to change a sentence from Lev Tolstoy's book to sound more like a liric from Nicki Minaj's song without changing the main
message of it; or if you want to add resistance to a certain environment to a developing drug.

In this project we try to formulate a universal framework for this kind of problems and test it on one of such tasks.

## Task

Here our task is morphing one letter to another, so that artistic style stays the same. For example we morph F->E, E->F, E->D, A->B and more.
We think this a good place to start, since a lot of transformations are quit intuitive (for example to make E out of F you just add 
an additional stick at the bottom) so it will be easy to perform a manual sanity-check. Plus it's much more objective then author-specific traits 
in a text or painting.

We use notMNIST dataset for training.

## Method

We train two neural nets: a VAE, that learns a hidden representation of letters and learns to reconstruct them, and a classifier,
that will give us a probability distribution of what letter is on the picture. 

Then we take a latent representation of an image we want to morph, and use gradient descent to change it, so that it's 
probability distribution leans more towards target letter.

This way our letter will gain more and more characterictics of a target one as it gets closer to the place where it's
representations cluster. And because optimization has started with a vector of our input image, the result will (hopefully)
still have it's inner structure.

This framework is quit universal, since your VAE can be buit on any type of data and you can optimize for any number 
of characteristics (you just set a desired probability distribution to be multi-modal or you use a scalar-target). 
As you can see, this methods lacks any amount of mathematical rigor or any kind of formalism really, but hell - who cares, if it works!
If it works...

## Results

The results are quit mixed. 
As you can see on theese pictures, system really does change input images to look more like a desired letter while leaving
charactiristics like width and ratios of the lines and overall style. 

![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/first_real.png)
![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/first.png)

F->E

![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/second_real.png)
![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/second.png)

F->E

![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/third_real.png)
![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/third.png)

E->B

![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/fourth_real.png)
![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/fourth.png)

E->D

![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/fifth_real.png)
![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/fifth.png)

E->F

![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/sixth_real.png)
![alt text](https://raw.githubusercontent.com/TheRussianFella/VecOptim/master/images/sixth.png)

A->B

But not all pictures were so lucky. Many weren't changed at all or were formed to monstrosities that shouldn't be discussed
in public. My main guess of why it happens is that classifier get's easily fooled, so the algorithm might work better
for a task where objectives can be analytically derived (or we might try building a better classifier).

## Summary

I think we have prooved that this method might work for certain domain of tasks but it is far from being a 42-like-solution.
I will continue working on Generative models and I hope to build something better for more complex object-modification tasks.

Stay tuned. 
