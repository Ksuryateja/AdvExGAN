# AdvExGAN
We are at a stage in Deep Learning where we can create or generate data using generative algorithms to meet the set conditions. DCGAN’s or even vanilla GAN’s try to achieve the data distribution of the  Discriminator D given that the vanilla GAN loss is employed.

We can make the DCGAN mimic the required distribution by tailoring the loss to our need i.e. we can manipulate the distribution that the generator G is trying to achieve by setting a loss.  

AdvExGAN is a DCGAN that generates the perturbations and  hence generates the adversarial example. AdvExGAN actually takes the original image as an input and outputs the perturbation.  



# LOSS :


# Discriminator Loss : 

# LD(θd,x,y) = −logD(x;θd)y

# Generator Loss :

# LG(x,y) = D(x + ε·G(x);θd)y + α·max(||G(x)||p,1−3)

Where
 D has a regular cross-entropy loss

G’s loss is the likelihood of the perturbed images and a Lp regularization term with a lower bound of 1−3.

α  is the normalization constant for perturbation (higher means encourage smaller perturbation)

Given an image, AdvExGAN produces a perturbation.

For LG, we use the likelihood of the perturbed images with respect to D’s loss function. 

This means that we do not consider the predictions made by D on the unperturbed images, only the true labels for the unperturbed images.

AdvExGAN generated the perturbations and these perturbations are combined with the original images and adversarial examples are created.

I have trained the AdvExGAN model both adversarially and non-adversarially.

The results are tabulated below

Here the attacker is the Generator from AdvExGAN that is trained on the mentioned model as Discriminator.

cells in red indicate a poor performance than FGSM and cells in green indicate a better performance than FGSM

![](/results/results1.png)

![](/results/results2.png)

![](/results/results3.png)

![](/results/results4.png)

# Adversarial inputs (whitebox attacks, non-adversarially trained)

GoogLeNet with epsilon values mentioned below the images.

![](/results/GoogLeNet.png)

VGG16 with epsilon values mentioned below the images.

![](/results/VGG16.png)

LeNet with epsilon values mentioned below the images.

![](/results/LeNet.png)


# Analysis

The adversarially trained models i.e. where both G and D were adversarially trained showed resilience towards whitebox attacks.

The AttackGAN performed well in all cases and fooled the target model considerably except in the case of adversarially trained G and D whitebox attacks.

I also found that adversarially training D with G makes it more robust to attacks from G and, to a lesser extent,to attacks fromFGSM.

For a deeper model like GoogLeNet as D, black-box attacks were as effective as white-box attacks.

For simpler models like LeNet FGSM always outperforms AttackGAN attacks.
