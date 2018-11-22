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

From September to November 2018, StarAi ran through a Deep Reinforcement Learning course at the Microsoft Reactor in central Sydney. For the course we developed a few world firsts, one of which was being able to render in Colaboratory. 

Developed by [William Xu](https://www.linkedin.com/in/william-xu-yuzhou/), our rendering solution makes use of PyVirtualDisplay, python-opengl, xvfb & the ffmpeg encoder libraries. 

With all the being said, lets get started.

## OpenAi Gym Colaboratory Rendering code.

First we need to install the relevant libraries to make rendering possible. In Colaboratory, install PyVirtualDisplay, python-opengl, xvfb & ffmpeg with the following code:

```python
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
```

Note that the "!" exclamation mark in the commands above is what is known as a "shell magic command" and allows us to make calls to the underlying Colaboratory virtual machine's shell. In this case we are making calls to the shell in order to be able to install a couple of libraries that are not preinstalled on Colaboratory - being **PyVirtualDisplay**, **python-opengl**, **xvfb** & **ffmpeg.**

The "> /dev/null 2>&1" part of the command just mutes the called commands outputs. Useful on Colaboratory. Trust me.

Next up we have to import the relevant libraries into Colaboratory to get rendering to work-let's do it:

```python

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import math
import glob
import io
import base64
from IPython.display import HTML

```

We now need to use PyvirtualDisplay to create a "virtual display" that we will send our rendered frames to. Ideally you would like this virtual display to be the same screen resolution of the Gym environment that you are using. Do this using the code below:

```python
#eg screen resolution 1400x900

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

```

In order to render in Colaboratory, William wrote two helper functions "show_video" & "wrap_env". Copy and paste these into cell blocks in order to get Colaboratory rendering.

```python

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    
```
    
```python
def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

```

Now, in your OpenAi gym code, where you would have usually declared what environment you are using we need to "wrap" that environment using the wrap_env function that we declared above.


```python
#Where ENV_NAME is the environment that are using from Gym, eg 'CartPole-v0'

env = wrap_env(gym.make(ENV_NAME)) #wrapping the env to render as a video

```

Don't forget to call env.render() at some point during the training phase of your algorithm so that Gym itself enters "render mode".

Finally, right at the end of your algorithm we need to call our second helper function "show_video" to show our stacked frames using this method & render our environment.

    
```python

show_video()

```

## Test Colaboratory Notebook  

This entire method is available in our [test Rendering Colaboratory Notebook here](), which renders a completely random agent in the **Pacman OpenAi Gym Environment.**


One final note on this method is since Google Virtual Machine's that run Colaboratory do not have physical screens or  actual rendering hardware -  we used xvfb to create a "virtual screen" on Colaboratory and then used IPythonDisplay to capture the rendered frames and save them as a .mp4 video to be shown in browser. This means that unfortunately you have to wait for your algorithm to finish it's training sequence before you can see how well it performed on the environment in question. This is usually no dramas however, if you were running Gym locally you would have to do this anyways.


That's it. If you decide to use this work, please referance it!


