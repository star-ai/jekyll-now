---
layout: post
title: Rendering OpenAi Gym in Google Colaboratory.
author: Paul Conyngham
---

###### By: Paul Steven Conyngham

Early this year (2018) Google introduced free GPUs to their machine learning tool "Colaboratory", making it the perfect platform for doing machine learning work or research.

![alt text](https://star-ai.github.io/images/GoogleColaboratoryGPU.png "Google's Machine Learning Tool: Colaboratory")


If you are looking at getting started with Reinforcement Learning however, you may have also heard of a tool released by OpenAi in 2016, called "OpenAi Gym". Gym gives you access to a library of training environments with standardized inputs & outputs, allowing your machine learning "agents" to control everything from Cartpoles to Space Invaders.

<img src="https://star-ai.github.io/images/openaigymenvs.gif" alt="OpenAi Gym" width="800" height="200"/>

<br/><br/>

Unfortunately if you are looking at learning reinforcement learning or even performing research, it is currently impossible to see your agents results "live" in your Colaboratory browser, until now.

From September to November 2018, StarAi ran through a Deep Reinforcement Learning course at the Microsoft Reactor in central Sydney. For the course we developed a few World firsts, one of which was being able to render in Colaboratory. 

Developed by [William Xu](https://www.linkedin.com/in/william-xu-yuzhou/), our rendering solution makes use of PyVirtualDisplay, python-opengl, xvfb & the ffmpeg encoder libraries. 

With all the being said, lets get started.

## Getting Started.

First we need to install the relevant libraries to make rendering possible. In Colaboratory, install PyVirtualDisplay, python-opengl, xvfb & ffmpeg with the following code:

```python
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
```

Note that the "!" exclamation mark in the commands above is what is known as a "shell magic command" and allows us to make calls to the underlying Colaboratory virtual machine's shell. In this case we are making calls to the shell in order to be able to install a couple of libraries that are not preinstalled on Colaboratory - being PyVirtualDisplay, python-opengl, xvfb & ffmpeg.


