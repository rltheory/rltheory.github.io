---
layout: default
title: About CMPUT 653 (OLD)
nav_order: 3
---

# CMPUT 653: Theoretical Foundations of Reinforcement Learning W2022

The purpose of this course is to let students acquire a solid understanding of the theoretical foundations of reinforcement learning, as well as to give students a glimpse on what theoretical research looks like in the context of computer science.
The topics will range from building up foundations (Markovian Decision Processes and the various special cases of it), to discussing solutions to the three core problem settings:
- [planning/simulation optimization](/lecture-notes/planning-in-mdps)
- [batch reinforcement learning](/lecture-notes/batch-rl), and
- [online reinforcement learning](/lecture-notes/online-rl)

In each of these settings, we cover key algorithmic challenges and the core ideas to address these. Specific topics, ideas and algorithms covered include, for each topic:

- complexity of planning/simulation optimization; large scale planning with function approximation;
- sample complexity of batch learning with and without function approximation;
- efficient online learning: the role (and limits) of optimism; scaling up with function approximation.

While we will explore connection to (some) deep RL methods, mainly seeking an answer to the question of when can we expect them to work well, the course will *not* focus on deep RL.

## Pre-requisites
Students taking the course are expected to have an understanding of basic probability, basics of concentration inequalities, linear algebra and convex optimization. This background is covered in Chapters 2, 3, 5, 7, 26, and 38 of the [Bandit Algorithms book](https://tor-lattimore.com/downloads/book/book.pdf). One very nice book that covers more, but is still highly recommended is [A Second Course in Probability Theory](http://people.bu.edu/pekoz/A_Second_Course_in_Probability-Ross-Pekoz.pdf). The book is available online and also in book format. Chapters 1, 3, 4, and 5 are most useful from here.

It will also be useful to recall foundations of mathematical analysis, such as completeness, metric spaces and alike, as we will start off with results that will require Banach's fixed point theorem. This is covered, for example, in Appendix A of [Csaba's "little" RL book](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf). The [wikipedia page](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) on Banach's fixed point theorem is not that bad either.

## Instruction Team
- [Csaba Szepesv&aacute;ri](https://sites.ualberta.ca/~szepesva)
- [Alex Ayoub](mailto:aayoub@ualberta.ca)
- [Vlad Tkachuk](mailto:vtkachuk@ualberta.ca)

## Lecture Time (recordings [here](/pages/lectures))
Monday and Wednesdays from 2:00 PM - 3:20 PM (MST).

The first three weeks will be online only and the format will be that of a flipped class. For the rest (Jan 25 and later), we aim for traditional, in-person lectures with chalk-board talks. The room is GSB 5-53

## eClass
We will use eclass for assignment submissions. The link to join eClass can be found [here](https://eclass.srv.ualberta.ca/course/view.php?id=76687).
We will not use eClass for announcements and discussions. For these we will use Slack.

## Slack Channel
We have a channel called `#cmput653-discussion-w2022` on the Amii slack to discuss all topics related to this course. This channel is open to anyone who is on Amii slack.
If you are already a part of the Amii slack, feel free to join this channel.
For discussions related to marking, assignment schedule, etc. we have a second channel `#cmput653-private-discussion-w2022`, which is by invitation only.
The TAs will add anyone who is taking the course for credit to these slack channels.
All announcements will be made on `#cmput653-discussion-w2022`.


## Google Meet Information
The google meet information will be posted on the slack channel and on eClass.
This is relevant up to the point when teaching becomes in-person.

## Grading Policies
Can be found on [eClass](https://eclass.srv.ualberta.ca/course/view.php?id=76687).

## Lectures Notes
Lecture notes of last year's class is available on this site. The lecture notes for this year's class starts from this, but may be modified.
The lecture notes serve as the required text for this course.

## Flipped Class
For the first three weeks, as mentioned above, we will follow a flipped class format:
Students coming to class are required to
- read the associated lecture notes and/or watch the lecture recordings
- prepare and vote on questions on the slack discussion channel

In class time will be spent on a
- quick review of the material
- discussing the most voted questions
- small group discussions of various topics

**Keywords:** RL theory, Reinforcement Learning, Theoretical Reinforcement Learning
