---
layout: post
title: What the Shell is a Multi Armed Bandit? - An introduction to Reinforcement Learning
author: Paul Conyngham
---

###### By: Paul Steven Conyngham

---

##### Guest starring the Epsilon-Greedy Algorithm, by Paul Steven Conyngham

---




![alt text](https://i.ibb.co/N9ff6tt/what-the-shell-by-paul.png)



Machine learning people love to give fancy names to things so that no one can understand what they are on about.

For someone relatively new to machine learning, "The multi armed bandit problem" sounds just like one of these fancy names.

Fret not however!

The point of this blog post is to explain exactly what a Bandit is and most importantly, why it is usually used as the starting point for anyone looking at learning Reinforcement Learning.

&nbsp;

&nbsp;

___

## Post Overview:

In this post I am going to aim to teach you:

1.  Some core Reinforcement Learning ideas such as the multi-armed bandit, exploration vs. exploitation & the epsilon greedy algorithm.
2. Introduce you to OpenAi gym and why it is important.
3. A programming exercise to help you solidify your understanding of the discussed ideas.

___

&nbsp;



&nbsp;



So then, what the shell is a bandit?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;









![alt text](https://i.ibb.co/xHx2jFt/Screen-Shot-2019-01-18-at-8-29-44-pm.png)

This.

A bandit is an old fashioned american name for what we usually call a "slot machine".

Here in Australia we like to call it the "pokies".

Great, but what the shell is a multi armed bandit?



&nbsp;

&nbsp;

&nbsp;

Simply, this:

![alt text](https://i.ibb.co/Cmm0NLn/Screen-Shot-2019-01-18-at-8-38-17-pm.png)

A whole bunch of "bandits" stacked together such there are many "arms" to pull.

This is where the "multi armed" part comes in.

&nbsp;

![alt text](https://i.ibb.co/8K8pbwx/multiararmedbandit-what-the-shell.png)

&nbsp;

Hold up.

Why are we talking about slot machines with regards to machine learning?

Well, one definition for Reinforcement Learning, the subfield of machine learning that we are talking about here deals with:

&nbsp;

>"Finding the optimal strategy to solving a problem in the face of massive uncertainty."

&nbsp;

Let's see why defining the multi-armed bandit this way is important by considering an example.

~

Say you wanted to drive your car from your home to your work.

When you wake up in the morning, you have no idea how the traffic lights will change or what the other cars will be doing on your way to work.

You could encounter 5 cars that delay you at a roundabout, or you could encounter 10 red traffic lights in a row.

In order to find this information out, you would have to actually drive the route to work and gather the data.

So, not knowing what the traffic will be doing *when you wake up in the morning*  is the uncertainty part in our definition above.



![alt text](https://i.ibb.co/Pr2wZ0t/Screen-Shot-2019-01-18-at-8-58-08-pm.png)



We also need a **_plan of what we will do_** once we encounter the intersections, roundabouts & traffic lights as we drive our car on our way to work.

This plan or **_strategy_** is what reinforcement learning aims to figure out.

Even more specifically, reinforcement learning attempts to learn the **_optimal strategy_** -  that is to say, the best possible strategy for a specific task - in this case, the best way of driving through all the obstacles of traffic lights, roundabouts etc from home to work.

This is the **_strategy_** part in our definition above.

&nbsp;

&nbsp;

___

###### Reinforcement Learning Terminology Decoded #1:

In reinforcement learning the name given to the strategy that we are following to solve a given problem is called a **_policy_**.

Following the previous example of driving to work. An example of a **_policy_** (strategy) would be driving as fast as possible to work.

Another example of a **_policy_** would be to ignore all red lights (again this could be another strategy).


Of course your policy could also be something boring. For example, only moving on green traffic lights, stopping for all other cars at roundabouts - like I hope you are doing.

Summarizing, **a policy** is the strategy that we use **to take incoming information** and **process it into actions to be taken in the environment.**

___

&nbsp;

&nbsp;

Let us watch a couple of people following the "policy" of not stopping at red traffic lights.

[![IMAGE ALT TEXT HERE](https://i.ibb.co/FDZXRzS/Screen-Shot-2019-02-11-at-8-43-00-pm.png)](https://www.youtube.com/watch?v=OJnZVjiscq4)

*Video 1: Here we can see the drivers following the policy of driving through the intersection when observing a red traffic light. :)*



Now we know that we are dealing with finding the best strategies -*or policies*- in the face of unknown conditions - *or uncertainty.*

The multi armed bandit problem is usually brought up as the starting point of most Reinforcement Learning (RL) text books because it introduces several core RL ideas.  

The first of which is that in slot machines - you are dealing with _uncertainty_.

In other words - if you go up to a slot machine and pull the lever, you have no idea ***when*** you are going to get a cash payout. You also have no idea ***how much*** that cash payout might be.

We also examine the multi armed bandit as our "toy" problem for explaining Reinforcement Learning because it teaches us the second core concept with regards to RL.

That is, what to do when we have more than one option for solving a problem.

In the multi-armed Bandit problem there are many slot machine levers to pull. So - we have many options. Just like you do when driving on your way to work.

How then do we decide upon the right strategy for how & when to pull the many different levers of our multi-armed bandit slot machine scenario, or in terms of our machine learning terminology from before - a “policy” for pulling these levers?

Summarizing, the bandit problem in a nutshell:

What do we do when we have more than one slot machine to choose from and we would like to know which slot machine is going to give us the highest average payout or **_reward_** over time. In other words, which slot machine is the best choice?

Lets get cracking on this problem by introducing another Machine Learning idea and your first (baby) Reinforcement Learning Algorithm.

&nbsp;

&nbsp;

&nbsp;


# Exploration vs Exploitation

There is this age old problem in life and it goes something like this.

![alt text](https://i.ibb.co/FqrfDc7/itsamatch-1.jpg)



Let’s say you are on the hunt for a hot date with the eventual goal of picking up a life partner.

First you hop on your favourite dating app of choice.

After chatting to a few people you manage to score yourself a hot date. The way they were coming on to you so strongly on the first date was a was a bit weird, strike out, you decide that's it for the dating app person.

The following week you decide dating apps are kind of lame, so you head out to the bar.

You manage to work up the courage to approach someone and after talking to them for a bit begin to realise that this person has really bad breath and might not be quite the right person for you.

Finally, towards the end of the night and after a few drinks, you work up your last bit of courage to go over and talk to an attractive person, who has been looking at you over their shoulder all night.

You hit it off and get their number. Two years later you are in a happy relationship and decide to get married, happily, ever after.

![alt text](https://i.ibb.co/dktFB3V/in-love-1071325-960-720.jpg)



What was just described above is an age old problem in machine learning.

How much time do you spend **"exploring"** - going on dates with people looking for a partner etc,

Versus,

How much time do you spend, ahem, **"exploiting"** - being in a relationship with someone etc.

&nbsp;

Lets make the topic of exploration vs exploitation more [concrete](https://www.youtube.com/watch?v=5ZNJPSe1nZs) with a few more examples.

Another example of exploration versus exploitation is how much time you spend looking at potential job opportunities (exploring) vs being in a particular job (exploiting).

![alt text](https://i.ibb.co/Tty4PWy/Screen-Shot-2019-01-18-at-9-47-26-pm.png)



Yet another example would be if you are a stock trader. How much time do you spend searching for the best trading strategy (exploring) vs implementing a strategy on the stock market (exploiting).

The reason we consider the exploration vs exploitation idea on the multi armed bandit problem, is that remember there is more than one machine and ***each slot machine's is tuned slightly differently***, such that each slot machine will give different average cash payouts.

**How then do we go about discovering which Bandit slot machine pays out the most?**

If you have not guessed it already, what we would like to do is spend some time **_exploring_** - to see amongst our many slot machines, which slot machine gives the best average payout.

When we have discovered which machine gives the best average payout, we then want to keep **_exploiting_** this machine.

Ok then. Time for your first RL (baby) algorithm

&nbsp;

&nbsp;

&nbsp;


# The Epsilon-Greedy Algorithm

Simply, the Episilon Greedy Algorithm is this:

![alt text](https://i.ibb.co/4TPp8h2/Screen-Shot-2019-01-19-at-1-02-05-pm.png)

Seriously though, if you did not understand that no dramas at all.

The rest of this post is going to be about breaking apart the mathematical notation above into a more human readable format :).

Let's do it.

&nbsp;

&nbsp;



------

##### Reinforcement Learning Terminology Decoded #2:

In machine learning we use another name for the **bell curve** (pictured below). That name is the "**normal distribution**". This is technically more correct as we will see soon.

Key idea: "A normal distribution is  "centered" around an average number."

![alt text](https://i.ibb.co/k3vDq8m/Screen-Shot-2019-01-21-at-7-55-09-am.png)

_Figure 2: Normal distribution centered around 100._

------

&nbsp;

&nbsp;

Say we wanted to implement some kind of **Exploration vs. Exploitation** on a problem.

How would we do it?

Let's examine the multi-armed bandit problem in a little bit more detail, to explain what **_epsilon_** & **_greedy_** is and why we are examining the multi armed bandit problem in the first place.

Say we have a bandit slot machine and that if I pulled the lever on the bandit 300 times, I would get data of the payout of that slot machine that looked like a bell curve like so:

![alt text](https://i.ibb.co/v4S6XKH/Screen-Shot-2019-01-21-at-7-50-39-am.png)

*Figure 1. Average payout of a slot machine based on the amount of times that the lever has been pulled. For example, we can see that at around the $70 mark the lever on the slot machine has been pulled around 44 times.*

A slot machine is meant to be **_random_** -  and we would like to discover a pattern in the “random” data if it exists.

We use the idea of a **_distribution_** to represent our Bandit's distribution of possible cash payouts. In essence the distribution is representing **_uncertainty_**.

Every time we pull our lever on the bandit, it will give a different cash payout.

In figure 1, sometimes it will be 65 dollars, sometimes it will be 80, but the average payout over time will be $70.

The middle of the bell graph in figure 1 is centered around the 70 dollar mark. We can therefore say that our slot machine in the graph in figure 1 has an average payout of around 70 dollars.

Let us now examine the case of two slot machines.

Each slot machine will each give payouts according to two **_different normal distributions._**

What does this mean? Have a look at the image below:



![alt text](https://i.ibb.co/1dsQtvS/Screen-Shot-2019-01-21-at-8-02-59-am.png)

_Figure 3, Here we can see that our first bandit, the bandit in blue,  which had an average value of  70 - is centered around the 70 dollar mark._

The bandit in pink in **bandit #2**.

We ***sample*** **Bandit #2** by pulling the lever on Bandit #2 over and over and collect new data on how Bandit #2 performs, seen in figure 3.

When plotted, we can see that bandit #2 has a ***different distribution*** and has a average payout centred around the $65 mark.

We have now introduced two bandits, bandit #1 & bandit #2.

We have also learned that each bandit has a different distribution.

Why is this important?

By “sampling” each bandit and building distributions, **we were able to determine which one of our bandits, on average, paid out the most money.**

The answer, if you have not guessed it already, is that Bandit #1 wins - with an average payout of 70 dollars as opposed to bandit #2, only paying out on average 65 dollars.

Linking back to what we talked about earlier, what we were doing by sampling each bandit and building distributions was **doing the exploration phase of the Epsilon Greedy algorithm.**

In summary,

Given a problem with many options.

What we would like to do is explore all the options available to us by randomly sampling between the different options, over and over until we start to build up a distribution of data for each option.

So, gather data on option 1 (Bandit #1), then a little bit of data on option 2 (Bandit #2) over and over until we have built distributions for all of our options.

Then, we calculate the average separately for each individual distribution we have gathered to discover the best option.

Once we have discovered the best option we can go ***"greedy"*** and continue to **exploit it.**

Let's now see where **Epsilon** comes in.



&nbsp;

&nbsp;

&nbsp;

# Epsilon

------

###### Reinforcement Learning Terminology Decoded #3:

According to wikipedia Epsilon is the 5th letter of the Greek alphabet and looks like this:
&nbsp;

# ε

&nbsp;

Hieroglyphics right? Epsilon is a greek letter. But what epsilon is used for is the interesting bit.

------

&nbsp;

**Epsilon-greedy is a mechanism used to decide which option to exploit.**

When ***sampling*** Epsilon controls the ratio between the amount of time we spend **exploring** vs how much time we spend **exploiting**.

Think of epsilon as a volume knob, which you can turn that controls the amount of exploration you do versus the amount of exploitation.

![alt text](https://i.ibb.co/9spLt4H/Screen-Shot-2019-02-11-at-7-33-20-am.png)

*Figure 4, Epsilon can be thought of a volume knob. Initially epsilon is high and exploration is maximised with a value of 10, but as time progresses we "turn" the volume knob down until it is low and exploitation is maximised having a value of 1. In the illustration above we are scaling from 10 to 1. In practise we usually scale from 1 to 0*

Initially we want to explore as much as possible to discover all the options available to us. To do this we set Epsilon to one.

So when ε = 1, exploration is maximised.

and when ε = 0, exploitation is maximised.

How then do we go from an epsilon of 1 down to 0?

Well, one way to do it is to choose a mathematical function to control Epsilon.

There are many mathematical functions you can use to control Epsilon. However in this example we are going control Epsilon using a linear function...or more commonly known as a straight line.

Think of our **linear function** **as the volume knob controlling the ratio between exploration and exploitation.**

&nbsp;

---

##### Here we derive a simple linear way of controlling Epsilon. Skip this part if you do not care about the maths.

&nbsp;

From high school mathematics, a straight line has the form:

(1.)   $y = mx + c ​$  

where m equals the gradient of the line, and c is the Y axis intercept.

![alt text](https://i.ibb.co/dPVsRNb/Screen-Shot-2019-02-11-at-7-37-38-am.png)

*Figure 5, A straight line plotted. Here the variable C (the Y axis intercept) has been set to zero.*

If we choose a straight line as our function for controlling Epsilon then our function f(x) becomes:

 (2.) $f(x) = mx + c $

We know we would like to start off with exploration maximised, so equal to one. We would also like to scale down Epsilon over time.  One way to control Epsilon would then be to subtract a straight line from one. Epsilon can then be defined as:

 (3.) $ε = 1 - f(x)​$

Substituing equation 2 yields:

 (4.) $ε = 1 - (mx + c) $

If we replace x for t (time) this gives us our final equation:

 (5.) $ε = 1 - mt - c ​$

Finally, if we set c to zero, we get the final diagram below. Exactly the type of operation that we are after- scaling epsilon from 1 to 0:

 (6.) $ε = 1 - mt $

![alt text](https://i.ibb.co/hm7nMtx/Screen-Shot-2019-02-11-at-7-39-19-am.png)

*Figure 6, Epsilon decreasing linearly over time.*

---

&nbsp;

So to summarize the epsilon-greedy process, via **sampling** we slowly start to figure out what is the best possible option to use to solve a problem. Simultaneously,  whilst we are sampling **we turn down the Epsilon volume knob.**

Turning down  Epsilon decreases the amount of exploration we are doing and starts focusing in on the best solution we have found so far to solve a given problem. Aka going **"greedy".**

Taking this back to our toy problem of bandits, we first start exploring by pulling different levers at random between Bandit # 1 and Bandit # 2.

Following this random sampling process, we start to build distributions of our different bandit options and see which bandit is going to give us the highest average reward in the form of cash payout.

***Simultanously as we are sampling, we start to decrease Epsilon and start focusing on the bandit option that is going to give us the highest average cash reward.***

In the video below I run though a visual example of the Epsilon-Greedy algorithm running in practise.

[![IMAGE ALT TEXT HERE](https://i.ibb.co/Bcf4GqH/Screen-Shot-2019-02-11-at-8-13-31-am.png)](https://youtu.be/bg65gXhYldE?t=395)

*Video 2: pulling it all together, lets run the epsilon greedy algorithm*

&nbsp;

&nbsp;

&nbsp;

# The final piece of the puzzle: the Reward-Averaging Sampling learning rule.

&nbsp;

In reinforcement learning we like to refer to our algorithm systems as "agents". So far we have learned about the "Epsilon-Greedy" agent.

We have seen that it is a synthesis of a purely exploratory agent and a completely greedy agent.

In the multiarmed bandit slot machine problem a ***purely exploratory agent*** would sample all the different bandits options available evenly - building a distribution for each bandit.

However this has the downside that the agent never gets to use its knowledge of the best option it has discovered so far.

One way to think of a ***purely exploratory agent*** is a student that goes to University and starts 5 different degrees - but quits after 1 semester in all of them - never sticking it out to see which degree might have been best for them, or in other words, giving them the highest average reward. ;)

However, ***A purely greedy agent***  would choose a bandit and stick to it's choice for all eternity.

A purely greedy agent can be though of as being very "narrow minded" as it will not try other bandit options to see if they provide better average long term reward.

To get the best of both worlds, the Epsilon-greedy agent is designed to ***explore*** at an ***Epsilon chance*** whilst the rest of the time it goes ***greedy*** on the best option it had discovered so far.

In our example, we have seen that we are able to scale the Epsilon chance using a linear function to control the amount of time we spend exploring versus the amount of time we spend going greedy.

The idea here is that the Greedy mechanism helps the agent exploit the best option it has discovered so far, whilst the small amount of exploration leftover ensures that our agent keeps searching the other options available - to prove that there are ***not even better options out there.***

The last remaining piece of the puzzle in the Epsilon-Greedy algorithm is how do we assign the idea of "value" to each of our different bandits options.

&nbsp;

&nbsp;

------

###### Reinforcement Learning Terminology Decoded #4:



### Q Values

Say we are at a single slot machine. We pull the slot machine's lever, and the shapes on the slot machines screen begin to whur! After a small amount of time the characters stop. We see "apple apple pear". Unfortunately, no money comes out of the machine.

In RL, we refer to the information on the slot machine's screen - "apple apple pear"- as ***State***.

The ***Action*** we took was pulling the lever.

The assignment of reward to this State-Action pair combo is what we call a ***Q-Value***. Q in this case just stands for Quality.

In our example we take an **action** and pull the lever on the slot machine. The information we get back as a result of this action is "apple apple pear" and we refer to this information as ***State***. The value of this State-Action pair is the resulting reward. So in this case our cash payout was zero. Therefore our Q value for this State-Action pair is 0.

The mathematical notation for assignment of a Q value looks like this:

$Q(s,a) = 0$

Where s & a represent state & action respectively.

On a final note the definition of Q values above is not quite the entire story, but we will get to why that is soon.

------

&nbsp;

&nbsp;

Let's say we pull the lever on one of our slot machine systems. We can assign a name to this action- lets call it A1.

The result from our action- A1- is that we see new information from the slot machine system and encounter a reward      - which we can call R1.

Given some state of the multi-slot machine bandit system we can specify a mathematical function to define the long term average reward. In Reinforcement Learning, we also like to call the *average* in this scenario ***"the expected value".***

Given many actions, or pulls of our lever on the bandit, the *expected* Q value or average over time for any one bandit choice, can be given by:

&nbsp;

##  $ Q_{n}(a) = \frac{1}{n}(r1 + r2 +r3)  .(1) $

&nbsp;

Where:

- **Q**  is our Q Value standing for **Quality**, this is an arbitary naming convention
- **n** stands for **the number of times that bandit (or state) has been visited**
- **r** is reward from visiting a bandit, n many times

&nbsp;

So for example, say we sample bandit # 1, 3 times.

This means that we get three different rewards r1, r2 & r3.

Because we visited bandit#1 3 times we also know that n = 3

Therefore:

## $ Q = (1/n)*(r1+r2+r3) .(2) $

&nbsp;

substituting 3 for n gives us:

&nbsp;

## $ Q = (1/3)*(r1+r2+r3) .(3) $

&nbsp;

*Equation 3:  the total expected reward or average, for a bandit option that has been "pulled" three times.*

&nbsp;

By pulling the lever, we are sampling our Bandit # 1,  3 times. As a result, we get 3 different rewards. We calculate the average reward for Bandit #1 and this gives us our Q value associated with that Bandit. We now know the "Quallity" of Bandit #1. Remember in machine learning in general we like to call the ***average*** the [***"total expected reward."***](https://en.wikipedia.org/wiki/Expected_value)

***We just worked out the average or expected value, for Bandit #1.***

Now lets say we wanted to keep track of the total expected reward of another bandit, say ***Bandit #2.*** In order to work out the average reward for **bandit #2,** we will have to keep track of all of **Bandit 2's** reward variables & number of times it has been visited in *a seperate table* to those of **bandit #1.**

Like so:

![alt text](https://i.ibb.co/4J1SmFq/Screen-Shot-2019-02-21-at-7-55-55-am.png)

*Table 1: Each Bandit has a seperate associated Q value, which we update individually each time that bandit is sampled.*

&nbsp;

In a real life multi armed bandit scenario, we would have to keep track of many reward variables the longer our agent is run.

For 8 "samples" of our bandit this would looks like this:

$ Q = (1/8)*(r1+r2+r3+r4+r5+r6+r7+r8) ... $

If we were to continue to run this algorithm for a while, you can quickly see that we would run out of computer memory.

In order to reduce the amount of memory that is in use, we can use some mathematical trickery to compress the above equation, such that when we update our Q value with every visit to the bandit our formula becomes:

## $ Q_{n+1}(a) = Q_{n}(a) + \frac{1}{n+1}(r_{n+1}-  Q_{n}(a))  .(4) $

Where:

- $ Q_{n+1}(a)  $ is the update to our Q table's value for a specific bandit, say bandit #1
- $ Q_{n}(a) $ is the current value in our Q table for a specific bandit
- **n +1** is the number of times that bandit (or state) has been visited. There is a "+1" here to stop a divide by zero error at the intialization of the algorithm.
- **r+1** is the updated reward from out latest sample of the bandit.

Finally, now that we have calculated our table of Q values, we need to select from our table the Bandit with the highest average payout.

Again, we are selecting the Bandit with the highest average payout as our "choice" amoungst the many options of Bandits that are available to us because it is the best choice we have discovered so far.

We do this by something that sounds like a super hero - using the argmax function.

Argmax stands for maximum argument and simpy put, means that given a choice of a whole bunch of numbers, say 1, 2, 3 - to choose the biggest number in the set, in this case the argmax of our set is 3.

If we were to take the argmax of our table 1 from before & substituting Q1=1, Q2=2 & Q3=3, then taking the argmax would just be the selection of the highest value in the table, the red square, like so:

&nbsp;

![alt text](https://i.ibb.co/QFKJ5rn/Screen-Shot-2019-02-22-at-8-12-37-am.png)

&nbsp;

If we continue to run our Epsilon Greedy agent for a while on a problem, say the multiarmed bandit, as discussed before the values of our bandit options in our Q table change. We then retake the argmax, and viola! once again find the biggest value in our set.

![alt text](https://i.ibb.co/tLNh3JK/Screen-Shot-2019-02-22-at-8-19-15-am.png)



The Reward-Averaging Sampling learning rule is then the heart of the epsilon greedy algorithm, or the forumula we use to update our Q table, **you have to use this code in the upcoming exercise to make the Epsilon-Greedy agent work.**

In summary, Epsilon Greedy is a powerful method for when you have a whole bunch of options, whose reward distribution you do not know & are able to find the optimal reward distribution using sampling.

The important thing to take away is that you can replace the Bandit problem with any problem which has different options which have hidden distributions. For example - an email marketing  campaign, and applying the exact same methods, determine the best products to display in that email campaign.

&nbsp;

&nbsp;

&nbsp;

# **Enter OpenAi Gym.**

___

> "What I cannot create, I do not understand" - Richard Feynman.

We are going to make extensive use of Gym in the accompanying programming exercise to this blog post. I highly suggest you have a crack at implementing Epsilon-Greedy on a multi-armed bandit scenario, as whilst I hope you may have gained the intuition of how the Multi Armed Bandit problem & Epsilon-Greedy Algorithm works in the explanation above - the best way to understand it is to build it yourself.

Only 3 years ago, OpenAi released Gym. There was little to no media coverage of the significance of this release.

It was actually really important.

Before OpenAi gym, there was no “standard” framework you could use test your Reinforcement learning algorithms on.

If a researcher in China was doing some work on Reinforcement Learning & wanted to have a benchmark to test it against a researcher's work in Australia - this was not possible, as both researchers in their respective countries were using different frameworks that they had written *themselves* to test their algorithms on.

In the accompanying exercise to this post below, we are going to make extensive use of Gym and implement an epsilon-greedy algorithm to solve the bandit problem.

Lets get cracking building your very first intro reinforcement learning algorithm with OpenAi gym.

We will be using Google Colaboratory for the compute, so you will also need a Google account.

To start the exercise [click the link here](https://colab.research.google.com/drive/1AgvnqbumrkPAFKI-Apt1SUtvbws4jVSS), or image below:

&nbsp;

[![IMAGE ALT TEXT HERE](https://i.ibb.co/52nm1gw/Screen-Shot-2019-02-22-at-9-16-59-pm.png)](https://colab.research.google.com/drive/1AgvnqbumrkPAFKI-Apt1SUtvbws4jVSS)
