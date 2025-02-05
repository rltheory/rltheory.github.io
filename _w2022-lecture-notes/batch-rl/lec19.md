---
layout: post
title:  19. Scaling with value function approximation
nav_order: 3
parent: Batch RL
publish_on_site: false
comments: true
---

[PDF Version](../../../documents/lectures/winter_2022/website_notes/batch_rl/lec19.pdf)

## Batch-mode Approximate Dynamic Programming

The next question is whether the "featurized" planning algorithms that use value function approximation can be adopted to the batch setting and if so, how?

For planning in the presence of simulators, three basic settings that are a good fit to "approximate dynamic programming" methods are summarized in the following table:

| Algorithm family | Condition | Specific algorithms |
|------------------|-----------|---------------------|
| AVI | $T \mathcal{F} \subset \mathcal{F}$ | LSVI/LSQI/FQI, DQN |
| API | $\forall \pi$: $q^\pi \in \mathcal{F}$ | LSPI, Politex, NGD |
| AMPI |$\forall \pi$: $T_\pi \mathcal{F} \subset \mathcal{F}$ | Actor-critic variants |

In this table $T$ stands for the Bellman optimality operator, $\mathcal{F}$ is the set of value functions that can be compactly represented (e.g., when using linear function approximation via the linear combination of a few basis functions).
Here, the domain of $T$ and $\mathcal{F}$ obviously need to match.
The condition $T\mathcal{F}\subset \mathcal{F}$ is known as the Bellman completeness condition. The condition in the last row is a stronger version of Bellman completeness. Indeed, this condition implies the one in the first row: Under the condition of the last row, if $f\in \mathcal{F}$ and $\pi$ is a greedy policy with respect to $f$ then $T f = T_\pi f$ and by the condition, $T_\pi f\in \mathcal{F}$, hence, $T \mathcal{F} \subset \mathcal{F}$ also holds.
<!-- Bellman rank, bilinear classes, etc. -->

In the first row, AVI stands for "approximate value iteration", in the second row API stands for "approximate policy iteration", while in the third row AMPI stands for "approximate modified policy iteration".
<!--
Scherrer, Bruno, Victor Gabillon, Mohammad Ghavamzadeh, and Matthieu Geist. 2012. “Approximate Modified Policy Iteration,” May. https://arxiv.org/abs/1205.3054v2.
TODO: Abbreviation war notes to the end.
-->

When a simulator is available, the algorithms are made to work by ensuring that the worst-case approximation error (or extrapolation error) is under control.
This is done by generating data using the simulator, usually from a set of carefully chosen "anchor points" (e.g., using an approximate G-optimal design).
When only a batch of data is available, choosing anchor points is not an option and this will obviously limit the extent to which the extrapolation error can be controlled. In fact, here, there is not much one can do: One can use all the data that is available, but nothing replaces data that is missing.

Still, there is a difference between how easily the various methods can be adopted to the batch setting.
The adoption of AVI and AMPI type methods that compute a sequence of value functions with the successive applications of a Bellman operator (either the Bellman optimality operator, or a policy evaluation operator) is relatively seamless. This is because these operators can be seen as the composition of an action selection operator ($M_\pi$ for policy evaluation and $M$, the greedifying operator for the Bellman optimality operator) and the transition kernel viewed as an operator. The action selection operator needs no approximation and the transition kernel operator can be just replaced with the empirical transition kernel. Finally, to keep the value functions in $\mathcal{F}$, a projection step is added.

In what follows we will discuss the first and the second row and the special demands that these come with in the batch setting.

## Batch-mode least-squares Q-iteration

For the sake of specificity, consider discounted problems and the case of approximating the Bellman optimality operator $T$ that maps action-value functions to action-value functions.
Thus,

$$
T q = r + \gamma P M q\,,
$$

where $M: \mathbb{R}^{\mathcal{S}\times \mathcal{A}}\to \mathbb{R}^{\mathcal{S}}$ is given by $(M q)(s)=\max_{a\in [A]} q(s,a)$ and, as usual, $\gamma$ is the discount factor.
Recall that value iteration then produces a sequence $q_k = T q_{k-1}$ and the policy returned is the policy that is greedy with respect to $q_K$ with some $K>0$.
When function approximation is involved, the relation $q_k = T q_{k-1}$ holds with some error:

$$
\begin{align}
q_k = T q_{k-1} + \varepsilon_k\,.
\label{eq:avi}
\end{align}
$$

Recall the following proposition from
an
<a href="/lecture-notes/planning-in-mdps/lec8#note:avi">
endnote</a> on approximate value iteration
from
[Lecture 8](/lecture-notes/planning-in-mdps/lec8/):

---
**Proposition (AVI error bound)**:
Let $q_0=0$. For any $K>0$,
the policy that is greedy with respect to $q_K$ is $\delta$-optimal

$$
\begin{align*}
\delta \le 2 H^2
\left(\gamma^K  + \max_{1\le k \le K} \| \varepsilon_k \|_\infty\right)\,.
\end{align*}
$$

---

This shows that in whatever approximate way we calculate $q_k$ from $q_{k-1}$, the algorithm's success will be controlled by the size of
$$\max_{1\le k \le K} \| \varepsilon_k \|_\infty$$.

### Least-squares approximation to $T$

In batch-mode least-squares Q-iteration these errors are controlled by performing a least-squares fit
from $\mathcal{F}\subset \mathbb{R}^{\mathcal{S}\times \mathcal{A}}$
 to a "noisy estimate" of $T q_{k-1}$, which is constructed on a batch
of transition data $D_n = ( (S_1,A_1,R_1,S_1'),\dots,(S_n,A_n,R_n,S_n'))$.
This data is assumed to be so that for $i\in [n]$, $(R_i,S_i')\sim Q(\cdot|S_i,A_i)$, with $Q$ the reward-state transition kernel of the MDP, as this, together with some assumptions on $Z_i:=(S_i,A_i)$ allows one to approximate $T$.
This is done as follows:
For $q\in \mathbb{R}^{\mathcal{S}\times\mathcal{A}}$, let   

$$
\hat T q = \arg\min_{f\in \mathcal{F}} L_n(f,q)\,,
$$

where

$$
L_n(f,q) = \frac1n \sum_{i=1}^n \left(R_i+\gamma \max_{a} q(S_i',a) - f(S_i,A_i) \right)^2\,.
$$

Defining $\hat P$ and $\hat r$ as the empirical state transition kernel and empirical reward function the usual way and the semi-norm $$\|f\|_n$$ through
$$\|f\|_n^2 = \frac1n \sum_{i=1}^n f^2(S_i,A_i)$$,
it is not hard to see that

$$\hat T q = \arg\min_{f\in \mathcal{F}} \| \hat r + \gamma \hat P M q - f \|_n^2,$$

or, as promised,

$$
\hat T q = \Pi_n (\hat r + \gamma \hat P M q),
$$

where $$\Pi_n g = \arg\min_{f\in \mathcal{F}} \| g - f \|_n^2$$.
With this notation, least-squares Q-iteration (LSQI) then computes the sequence

$$
q_{k+1} = \hat T q_k\,.
$$

When $\mathcal{F} = \{ \Phi \theta\,:\, \theta\in \mathbb{R}^d \}$, letting $q_n = \Phi \theta_n$, one can perform the whole calculation through calculating the parameter sequence $(\theta_n)_n$, where each update costs the same as finding the solution of a least-squares problem. If $\mathcal{F}$ is the set of functions that can be represented through neural networks and instead of calculating the minimizer of the least-squares problems, only a few gradient steps are made by also subsampling the data, one gets a variant of the "deep Q-networks" (DQN) method.
<!-- LSVI ~ LSQI -->
<!-- regularization? -->
<!-- homework to write the operators for AMPI -->

From our perspective, the main question is how good these methods can be? As it stands, the main issue is to control the worst-case error. In the linear case, under Bellman completeness, this (essentially)
hinges upon how the eigenvalues of the expectation of the moment matrix,

$$
G_n = \frac1n \sum_{i=1}^n \phi(S_i,A_i) \phi(S_i,A_i)^\top,
$$

behave.

To see how this works consider a simple case when $(S_i,A_i,R_i,S_i')$ is an i.i.d. sequence.
This is similar to the $Z$-design case, but this excludes the possibility when the data is serially correlated, which would be the case when it is generated by following some policy.
<!-- return to the single data path case later -->
To simplify notation let $Z_i = (S_i,A_i)\in \mathcal{Z}:=\mathcal{S}\times \mathcal{A}$ and $\phi_i = \phi(Z_i)$.
Fix $q\in \mathbb{R}^{\mathcal{Z}}$.
Assume further that $T q = \Phi \theta$ (realizability).
Letting $\epsilon_i = R_i + \gamma \max_a q(S_i',a) - \phi_i^\top \theta$, we see that $(\epsilon_i)_i$ is a zero-mean i.i.d. sequence.
Then, $\Delta:=\hat \theta_n-\theta = G_n^{-1} \frac1n \sum_i \phi_i \epsilon_i$
and

$$
\begin{align*}
\mathbb{E}[ \Delta \Delta^\top | Z_1, \dots, Z_n ]
& =
\frac{1}{n^2}
G_n^{-1}
\mathbb{E}[ \sum_{i,j} \phi_i \phi_j^\top \epsilon_i \epsilon_j | Z_1, \dots, Z_n ] G_n^{-1} \\
& =
\frac{\sigma^2}{n^2}
G_n^{-1}
\mathbb{E}[ \sum_{i} \phi_i \phi_i^\top | Z_1, \dots, Z_n ] G_n^{-1} \\
& =
\frac{\sigma^2}{n}
G_n^{-1} G_n G_n^{-1} \\
& =
\frac{\sigma^2}{n} G_n^{-1} \,,
\end{align*}
$$

where $\sigma^2 = \mathbb{E}[\epsilon_1^2|Z_1]$.
For $n$ large, we also have $G_n \approx G:=\mathbb{E}[ \phi(Z_1)\phi(Z_1)^\top ]$.
Hence, the squared expected error at $z$ satisfies

$$
\begin{align*}
\mathbb{E}[ (\phi(z)^\top \Delta)^2 ]
=
\phi(z)^\top \mathbb{E}[
\mathbb{E}[ \Delta \Delta^\top  | Z_1,\dots,Z_n ]
] \phi(z)
\approx
\frac{\mathbb{E}\sigma^2}{n} \| \phi(z) \|^2_{G^{-1}} \,.
\end{align*}
$$

(A similar calculation was done earlier in
[Lecture 8](/lecture-notes/planning-in-mdps/lec8/).)
Thus, the extrapolation errors are governed by the value of

$$
\epsilon(\Phi):=\sup_{z\in \mathcal{Z}} \| \phi(z) \|_{G^{-1}}\,.
$$

Note that this, in turn, is controlled by the common distribution of $$(Z_i)_i$$.
Recalling what we learned about $$G$$-optimal designs in the lecture mentioned previously, we know that there is always a distribution such that this is at most $\sqrt{d}$. However, if the distribution is not chosen in a careful manner then this value can be arbitrarily large. Indeed, to start with the the most significant issue, $G$ may even be singular. But even if $G$ is invertible, this value can be as large as $$\lambda^{-1/2}_{\min}(G)$$, the square root of the inverse of the minimum eigenvalue of $G$.
<!-- homework! -->
Altogether, this discussion shows that if the feature-space is a good fit to the MDP and the data generating distribution is favorable,
the errors
$\varepsilon_k = q_k - T q_{k-1}$
will be under control. While the above discussion was developed for the realizable case (when $T\mathcal{F}\subset \mathcal{F}$), the same conclusions hold for the misspecified case, as well.
The problem is that of course batch mode RL gives little room for controlling $\epsilon(\Phi)$ as in general the distribution $P$ of $(\phi(Z_i))_i$ is uncontrolled. When there is a chance to control this distribution, this discussion shows that it suffices to do it so that the minimum eigenvalue of $G$ is controlled (equivalently, $\epsilon(\Phi)$ is controlled).

To get a sense of the quality of the bound before moving to the next topic, it is worthwhile to consider the tabular case, which is a special case of the linear case, as noted beforehand (choose $d=SA$, and let the feature vector of each state-action pair be a unit vector of $\mathbb{R}^d$).
In this case
the choice for $P$ that maximizes the minimum eigenvalue is
the uniform distribution over the state action space. With this choice, the minimum eigenvalue becomes $1/d=1/(SA)$. Choosing $K$ so that $H^2 \gamma^K\le \delta/2$ (i.e., $K \approx H_{\gamma,\delta/(2H)})$), ignoring logarithmic terms,
we get that
it suffices to have a dataset of size $n=(H SA/\delta)^2$ to obtain a $\delta$-optimal policy. While this is worse than the best bound we obtained for the plug-in method, it is comparable to the bounds we obtained with simpler methods. In fact, with some conditions, the procedure itself is optimal; the suboptimal results are obtained because of the simple analysis.
<!-- Discussion: Correlated data represents not a big problem -->

## Batch-mode least-squares policy iteration

In policy iteration, $\pi_k$ is chosen to be greedy with respect to
the value function of the previous policy $\pi_{k-1}$.
Letting $q_k = q^{\pi_k}$, this means that $\pi_k$ satisfies
$M q_{k-1} = M_{\pi_k} q_{k-1}$. Again, a previous result in
[Lecture 8](/lecture-notes/planning-in-mdps/lec8/)
states that even if

$$
q_k = q^{\pi_k}+\varepsilon_k
$$

where $\varepsilon_k$ may be nonzero, this process is stable in the sense that the policy computed after $K$ steps is near-optimal as long as $K$ is large enough and the error
$$\varepsilon_{1:k}=\max_{1\le k \le K} \| \varepsilon_k \|_\infty$$ is under control. In particular, a <a href="/lecture-notes/planning-in-mdps/lec8#cor:apiq">corollary</a> in that lecture states the following:

---
**Proposition**:
Let $$(\pi_k)_{k\ge 0}$$, $$(\varepsilon_k)_k$$ be as above.
Then, for any $$k\ge 1$$,

$$
\| v^* - v^{\pi_k} \|_\infty
\leq H \gamma^k + H^2 \varepsilon_{1:k}\,,
$$

where $H=1/(1-\gamma)$.

---

Now, to control the suboptimality of the policy it suffices to choose $k$ large enough, while controlling the error $\varepsilon_{1:k}$.
In the lecture on planning, these errors were controlled by rolling out with the policies $\pi_i$ obtained in the process.
Clearly, when working with a batch of data, this option is not available.
As it turns out, evaluating **arbitrary** policies based on a batch of data can be challenging exactly because of this: How can we use data generated by following some policy (or a $Z$-design) to evaluate some other policy?
In what follows, we will enumerate some options for doing this and discuss their pros and const.

## Policy evaluation using likelihood ratio corrections

One simple idea that is applicable when the data is generated by following some logging policy is to use likelihood ratio corrections together with least-squares fitting.
In particular, assume that the data consists of a number of trajectories
generated by following policy $\pi_{\log}$ while our aim is to evaluate a policy $\pi$.
Let the initial state-action distribution be $\mu$.

As discussed in the part on planning, there is not much loss in assuming that we cut the trajectories after some $H$ steps with $H \approx H_{\gamma, \delta/(1-\gamma)^2}$. The expected total discounted reward under a policy along a trajectory of this length that is started from $\mu$ is then

$$
q^\pi_H(\mu):=\int R(\tau) P_{\mu}^{\pi}(d\tau)\,,
$$

where $P_{sa}^\pi$ is the distribution induced over the $H$ step-trajectories $\tau\in (\mathcal{S}\times\mathcal{A})^H$ by the interconnection of policy $\pi$ and MDP $M$ when the policy is started from $(s,a)$,
while $R(\tau)$ is the discounted sum of immediate expected rewards along $\tau = ((s_0,a_0),\dots,(s_{H-1},a_{H-1}))$:

$$
R(\tau) = \sum_{h=0}^{H-1} r(s_h,a_h)\,.
$$

For discrete state-action spaces, the above integral reduces to a sum
and becomes

$$
\sum_{\tau\in \mathcal{Z}^H} R(\tau) p_{\mu}^{\pi}(\tau)\,,
$$

where $p_{\mu}^\pi(\tau)$ is the probability of trajectory $\tau$.
In particular, if
$\tau = ((s_0,a_0),\dots,(s_{H-1},a_{H-1}))$ then

$$
p_{\mu}^\pi(\tau) = \mu(s_0,a_0) P_{a_0}(s_0,s_1) \pi(a_1|s_1) \dots
P_{a_{H-2}}(s_{H-2},s_{H-1}) \pi(a_{H-1}|s_{H-1})\,.
$$

In any case, for any $\pi'$, as long as the support of
$dP_{\mu}^{\pi}$ is at least as large as that of
$dP_{\mu}^{\pi'}$,
by a change of measure argument, we have

$$
q^{\pi'}_H(\mu) =
\int R(\tau) P_{\mu}^{\pi'}(d\tau)
=
\int R(\tau) \frac{dP_{\mu}^{\pi'}}{dP_{\mu}^{\pi}}(\tau)
dP_{\mu}^{\pi}(d\tau)\,.
$$

Here,
$
\rho:=\frac{dP_{\mu}^{\pi'}}{dP_{\mu}^{\pi'}}
$
denotes the density of measure $P_{\mu}^{\pi'}$ with respect to
$P_{\mu}^{\pi}$ (this is also known as the Radon-Nikodym derivative of
$P_{\mu}^{\pi'}$ with respect to $P_{\mu}^{\pi}$). (The definition of $\rho$ is that it is a measurable function over the trajectories of required length so that
    for any measurable subset $U$ of these trajectories,
$\int_U \rho(\tau) P_{\mu}^{\pi}(d\tau) = \int_U P_{\mu}^{\pi'}(d\tau)$).
The function $\rho$ is also known
as the **likelihood ratio**
of measure $P_{\mu}^{\pi'}$ relative to $P_{\mu}^{\pi}$
at $\tau=(s_0,a_0,\dots,s_{H-1},a_{H-1})$. Note that here,
for the sake of minimizing clutter, by slightly abusing notation,
we dropped some parentheses.

As it turns $\rho$ is simple to calculate and satisfies

$$
\begin{align}
\rho(\tau):=
\frac{dP_{\mu}^{\pi'}}{dP_{\mu}^{\pi'}}(s_0,a_0,\dots,s_{H-1},a_{H-1})
=
\prod_{h=1}^{H-1} \frac{\pi'(a_h|s_h)}{\pi(a_h|s_h)}\,.
\label{eq:rhofinite}
\end{align}
$$

From this, we see that the support condition holds if for any $(s,a)$ such that $\pi'(a\vert s)>0$, $\pi(a\vert s)>0$ also holds. Intuitively, the data generating policy should never omit actions that the **target policy** $\pi'$ would use with positive probability.

The above relation for $\rho$ is easy to verify from the definition above when the state-action space is discrete.
The important point here is that **evaluating $\rho$ at any trajectory requires only knowing $\pi$ and $\pi'$** and in particular the knowledge of the transition probabilities is not needed.
<!-- Monte-Carlo literature etc. -->

It follows then that $R(\tau)\rho(\tau)$ can be used as a target to estimate $q^{\pi'}_H$.
With linear function approximation this works as follows:
<!-- comment on nonlinear FAPP -->
Given $\mathcal{F} = \{ \Phi \theta \,:\, \theta\in \mathbb{R}^d \}\subset \mathbb{R}^{\mathcal{S}\times \mathcal{A}}$,
one can estimate
the "projection" $\Pi_mu q^{\pi'}_H$ of $q^{\pi'}_H$ to $\mathcal{F}$
using least-squares fitting based on a number of independently obtained trajectories $\tau^{(1)},\dots,\tau^{(n)}$.
Here, the projection mentioned is defined by

$$
\Pi_\mu q = \arg\min_{g\in \mathcal{F}} \int (g(z)-q(z))^2 \mu(dz)\,,
$$

which we assume is uniquely defined. In particular, this holds if $G= \int \phi(z)\phi(z)^\top \mu(dz)$ is full rank.

The estimate of $\Pi_{\mu} q^{\pi'}_H$ is then defined as $g_n=\Phi \theta_n$, where

$$
\begin{align}
\theta_n = \arg\min_{\theta} \sum_{i=1}^n ( R(\tau^{(i)})\rho(\tau^{(i)})- \phi(\tau^{(i)}_0) \theta )^2\,,
\label{eq:lspibtrg}
\end{align}
$$

where $\tau^{(i)}_0$ denotes the first state-action pair along trajectory $\tau^{(i)}$. Indeed, with some calculations similar to the one done previously, it is not hard to see that under minimal regularity assumptions,

$$
\| g_n - \Pi_{\mu} q_H^{\pi'} \|_\infty \lesssim \frac{\sigma}{n} \lambda_{\min}^{-1/2}(G),
$$

where $\sigma^2 = \mathbb{E}[ \sigma^2(\tau^{(1)}_0) ]$, with the  quantity inside the expectation defined by

$$
\sigma^2(z) = \mathbb{E}[ (R(\tau^{(i)})\rho(\tau^{(i)}) - q^{\pi}_H(\tau^{(i)}_0) )^2  | \tau^{(1)}_0 = z ]
= \mathrm{Var}[ R(\tau^{(i)})\rho(\tau^{(i)}) | \tau^{(1)}_0 = z ]\,.
$$

The problem now is twofold: On the one hand one needs to control the minimum eigenvalue of $G$ (the same optimal design problem that we met multiple times before), while, on the other hand, one also needs to control the variance terms $\sigma^2(z)$. While the rewards along the trajectories are bounded, that is, the range of $R$ above is bounded, this may scale with $H$ (in the lack of access to the immediate reward function, but when the trajectory data includes rewards along the transitions observed, one can of course replace $R(\tau^{(i)})$ with the sum of rewards in $\tau^{(i)}$, at the price of potentially further increasing the variance).

A potentially even more serious problem though is that the variance of the likelihood ratios can be quite large.
Indeed,
owning to $\mathbb{E}[\rho(\tau)]=1$,
$\mathrm{Var}(\rho(\tau))=\mathbb{E}[\rho^2(\tau)]-1$.

Then, the Perron-Frobenius theory of nonnegative matrices implies the following:

---
**Proposition**:
Assume that there are finitely many state-action pairs and
the Markov chain over the state-action pairs induced by following $\pi'$ is ergodic (aperiodic and irreducible). Assume further that $\pi'(a|s)>0$ implies that $\pi(a|s)>0$.
Then,

$$
\begin{align}
\mathbb{E}[\rho^2(\tau)]
\sim C e^{H \alpha}\,,
\label{eq:expasymptotics}
\end{align}
$$

where the asymptotics is taken with $H\to \infty$,
$C>0$ is a positive constant and $\alpha>0$ is the largest magnitude eigenvalue of the $SA \times SA$ matrix $F$ whose entry at $(s,a)$ and $(s',a')$ is zero if $\pi(a'|s')=0$ and otherwise it is

$$
P_{a}(s,s')\pi(a'|s')\left(\frac{\pi'(a'|s')}{\pi(a'|s')}\right)^2
$$

---

**Proof**:
Let $g(s,a)=(\frac{\pi'(a|s)}{\pi(a|s)})^2$.
Note that $\rho(\tau) = g(Z_1)g(Z_2)\dots g(Z_{H-1})$ where $Z_i = (S_i,A_i)$.

For $n\ge 1$, define the $SA \times SA$ matrix $F(n)$ by

$$
\begin{align*}
F_{z,z'}(n)
& =\mathbb{E}_{z}[g(Z_n)g(Z_{n-1})\dots g(Z_0) \mathbb{I}(Z_n=z')]\,,
\end{align*}
$$

where $Z_0,Z_1,\dots$ are the state-action pairs of the Markov chain on $\mathcal{Z}:=\mathcal{S}\times\mathcal{A}$ induced by $\pi$ and $\mathbb{E}_z$ is the expectation under the distribution of this chain when the chain starts at $z$.
It follows from elementary calculations that for any $n\ge 1$,
$F(n) = F^n$. We prove this by induction on $n$.
Indeed, this holds for $n=1$ by inspecting the definitions. Now, assuming that $F(n)=F^n$ holds for some $n\ge 1$, we have

$$
\begin{align*}
F_{z,z'}(n+1)
& = \mathbb{E}_{z}[g(Z_{n+1})g(Z_n)\dots g(Z_0) \mathbb{I}(Z_{n+1}=z')]\\
& = \sum_u \mathbb{E}_{z}\left[
\mathbb{E}_{z}[g(Z_{n+1})g(Z_n)\dots g(Z_0) \mathbb{I}(Z_n=u,Z_{n+1}=z')|Z_0,\dots,Z_{n}]\right] \\
& = \sum_u \mathbb{E}_{z}\left[
g(Z_n)\dots g(Z_0) \mathbb{I}(Z_n=u)
\mathbb{E}_{z}[g(Z_{n+1}) \mathbb{I}(Z_{n+1}=z')|Z_{n}]\right] \\
& = \sum_u \mathbb{E}_{z}[
g(Z_n)\dots g(Z_0) \mathbb{I}(Z_n=u)
F_{Z_n,z'}] \\
& = \sum_u F_{z,u}(n) F_{u,z'}  = (F^{n+1})_{z,z'}\,,
\end{align*}
$$

where the last equality used the induction hypothesis. Now, since $F$ is nonnegative valued, by
<a href="https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem">Perron-Frobenius theory</a>,
$F^n \sim h \kappa e^{n \alpha}$ where $\kappa$ is a left-eigenvector of $F$ and $h$ is a right-eigenvector of $F$, both corresponding to the eigenvalue $\rho$ and they are normalized so that $\kappa h=1$ <!-- and $\nu^* h =1$ where $\nu^*$ is the unique stationary distribution of $(Z_n)_n$ -->
($h\kappa$ is called the Perron projection underlying $F$).
The result follows by noting that
$\mathbb{E}[\rho^2(\tau)]=\mu F(H-1) \boldsymbol{1} \sim \mu h \kappa \boldsymbol{1}e^{\alpha}\, e^{\alpha H}$.
The constant $C=\mu h \kappa \boldsymbol{1} e^{\alpha}$ is positive since $F$ is irreducible, hence both $h$ and $\kappa$ have only positive entries.
$$\qquad \blacksquare$$

We can further conclude that the growth rate of the variance is rapid. In particular, $\alpha \ge 1$. This follows, since also from Perron-Frobenius theory, we have

$$
\alpha\ge \min_{s,a} (F \boldsymbol{1})_{s,a}.
$$

Now, from Jensen's inequality,

$$
\begin{align*}
(F \boldsymbol{1})_{s,a}
& = \sum_{s',a':\pi(a'|s')>0}
P_{a}(s,s')\pi(a'|s')\left(\frac{\pi'(a'|s')}{\pi(a'|s')}\right)^2
 \ge
\left(
\sum_{s',a':\pi(a'|s')>0}
P_{a}(s,s')\pi(a'|s')\frac{\pi'(a'|s')}{\pi(a'|s')}\right)^2\\
& =
\left(
\sum_{s',a':\pi(a'|s')>0}
P_{a}(s,s')\pi'(a'|s')\right)^2 = 1\,.
\end{align*}
$$

<!-- https://web.stanford.edu/~glynn/papers/1987/G87c.pdf
https://www.informs-sim.org/wsc87papers/1987_0053.pdf
1987 Likelihood ratio gradient estimation: An overview
Peter W. Glynn, Winter simulation conference.
Bad calculation!?
Better:
Prop 5.5 in Chapter XIV.  of
Asmussen, Søren, and Peter W. Glynn. 2007. Stochastic Simulation: Algorithms and Analysis. Vol. 57. Stochastic Modelling and Applied Probability. New York, NY: Springer New York.
who uses a result from
  S. Asmussen (2003)Applied Probability and Queues(2nd ed.). Springer-Verlag.
-->
There is a good intuitive explanation as well as to why the variance of $\rho$ grows really fast.
Since the data is sampled from $\pi$, we expect the ratios in $\rho$ to be less than one: It is rare to sample an action from $\pi$ which has a probability under $\pi$ that is below the probability assigned to the action by $\pi'$: sampling under $\pi$ chooses actions which have high probability under $\pi$. Thus, the typical terms in $\rho$ are below 1. In fact, because of this, one can show that with probability one $\rho$, as $H\to\infty$, converges to $0$. Now, since the expectation of $\rho$ is one, there must be exponentially rare events when $\rho$ takes on exponentially large values.
<!-- refine this discussion -->

## Variance reduction

There is a trivial approach to reduce variance, which is based on the observation that $R(\tau)$ has an additive structure and that
for a given $h$ index, the state-action pairs beyond index $h$ have no influence on the expectation of $r(S_h,A_h)\rho(\tau)$.
In particular, we have that

$$
\mathbb{E}[ r(S_h,A_h)\rho_{H-1}(\tau) ]
=
\mathbb{E}[ r(S_h,A_h) \rho_h(\tau_h) ]\,,
$$

where $\tau_h = (S_0,A_0,\dots,S_h,A_h)$ and

$$
\rho_h = \frac{dP_{h,\mu}^{\pi'}}{dP_{h,\mu}^{\pi}}\,,
$$

where $P_{h,\mu}^\pi$ denotes the distribution induced
over trajectories of length $h+1$
 by using $\pi$ in MDP $M$ starting from $\mu$.
<!-- homework to prove the above -->
Clearly, we still have

$$
\rho_h(s_0,a_0,\dots,s_h,a_h) =
\frac{\pi'(a_1|s_1)}{\pi(a_1|s_1)}
\dots
\frac{\pi'(a_h|s_h)}{\pi(a_h|s_h)} \,.
$$

One can then replace $R(\tau^{(i)})\rho(\tau^{(i)})$ in \eqref{eq:lspibtrg}
by

$$
r(S_0^{(i)},A_0^{(i)})+\sum_{h=1}^{H-1}
r(S_h^{(i)},A_h^{(i)})
\rho_h(S_0^{(i)},A_0^{(i)},\dots,S_h^{(i)},A_h^{(i)})
$$

where $\tau^{(i)} = (S_0^{(i)},A_0^{(i)},\dots,S_{H-1}^{(i)},A_{H-1}^{(i)})$.

While this generally has lower variance than the previous expression, the fundamental problem remains.
The likelihood ratio method is a general technique in the Monte-Carlo simulation literature. As such, there is a whole range of further ideas that can be applied to reduce the variance even further.
Examples include baseline substraction, the doubly robust estimator, self-normalized importance weighting (the likelihood ratio in this context is also known as importance weighting), etc.
However, rather than embarking on the futile mission of trying to survey all these and the other methods, we move on and consider the last case in our table.

## TD methods and variants

When $T_\pi \mathcal{F}\subset \mathcal{F}$ for the policy $\pi$ to be evaluated, one can follow the approach 
