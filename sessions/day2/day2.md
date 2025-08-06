# Day 2: ML Overviews, Differentiable Programing and NSBI

## Neural simulation-based inference (NSBI)

In HEP processes, both the absolute rate, $\sigma_H$, and distribution, $p(x | H)$, of a process/hypothesis, $H$, are encoded in the squared matrix-element squared, $|\mathcal{M}(z ; H)|^2 $.
The ME is fundamentally a function of *latent* variables, $z$, which cannot be empirically observed, such as the partonic four-momenta.
In a realistic LHC environment, these partonic interactions occur inside $pp$ bunch crossings and thus undergo various non-perturbative processes accounting for PDFs, parton showering, and hadronization.
Finally, particle detectors such as ATLAS and CMS reconstruct the events, up to acceptance/efficiency/resolution effects. 
Additional latent variables from these processes enter the description.
Only after all of these effects are accounted for can the rate of events be measured differentially as a function of *observables*, $x$.
This process by which the likelihood is rendered intractable can be expressed as a convolution over the latent space to the observable space:

$$ \frac{d\sigma_H}{dx} = \sigma p(x | H) = \int dz \; p(x | z) |\mathcal{M}(z ; H)|^2 $$

In general, the likelihood post-convolution, $p(x | H)$, cannot be numerically evaluated, i.e. is rendered *intractable*.
Instead, one can generate samples of events for processes of interest via *simulators* that can model these complex processes in a forward mode:

$$
x \sim p(x | H)
$$

The goal of neural simulation-based inference, in a frequentist setting, is to train a class of neural networks that learn from these generated samples to estimate the ratio of probability density (DRE) of an event between two hypotheses:

$$
r (x ; H_1, H_2) \equiv \frac{p(x | H_1)}{p(x | H_2)}.
$$

Henceforth, $H_1$ and $H_2$ are referred to as _numerator_ and _denominator_ hypotheses, respectively.

In this tutorial, we will be implementing NSBI to establish confidence interval on a single paramter of interest.
The tutorial consists of individual chapters covering:

1. Data exploration: phenomenology of the $gg \to (h^{\ast} \to) ZZ \to 4\ell$ process.
2. Training classifiers that approximate ratios of the likelihood ratio (CARL).
3. Evaluating the accuracy of the density ratio estimates via diagnostics.
4. Using the CARL estimates to perform neural simulation-based inference of the Higgs signal strength.

As soon as your environment is setup, your are strongly encouraged to go ahead and launch the NN training, which is already available in `scripts/fit-carl.sh`. So that by the time you digest the material in chapters 1 & 2 (approx. 15 minutes each), you are ready to use the models for 3 & 4.

### Physics at hand: Off-shell Higgs production

The physics problem at hand to be studied is the off-shell production of the Higgs boson. Disregarding the $q\bar{q}$-initiated processes (imagine an LGC: the Large Gluon Collider), the leading-order $gg \to 4\ell$ diagrams are:

1. Higgs signal via gluon-fusion production, $gg \to h^{\ast} \to ZZ \to 4\ell$.
2. Continuum background via box diagram, $gg \to ZZ \to 4\ell$.

```image ./gghzz.png
:width: 300px
:align: center
```
```image ./ggzz.png
:width: 300px
:align: center
```

These processes non-trivially interfere with each other: in other words, the full matrix-element squared of the $gg\to (h^{\ast}\to) \to ZZ 4\ell$ is given by:

$$
\left| \mathcal{M}_{\rm S} + \mathcal{M}_{\rm B} \right|^2 = |\mathcal{M}_{\rm S}|^2 + 2 \mathbb{R} ( \mathcal{M}^{\dag}_{\rm S} \mathcal{M}_{\rm B} ) + \left| \mathcal{M}_{\rm B} \right|^2.
$$

BSM mdifications to the Higgs signal process is assumed to occur through the signal strength parameter, $\mu \in \mathbb{R}$, that scales the signal-squared & inteference terms as:

$$
\left| \sqrt{\mu} \mathcal{M}_{\rm S} + \mathcal{M}_{\rm B} \right|^2 = \mu |\mathcal{M}_{\rm S}|^2 + \sqrt{\mu} 2 \mathbb{R} ( \mathcal{M}^{\dag}_{\rm S} \mathcal{M}_{\rm B} ) + \left| \mathcal{M}_{\rm B} \right|^2.
$$

In terms of probabilities, the mixture model of the full signal+background+inteference (SBI*) hypothesis as a function of the signal strength parameter becomes

$$
\frac{p_{\rm SBI} (x | \mu)}{p_{\rm B} (x)} = \frac{ (\mu - \sqrt{\mu}) \sigma_{\rm S} p_{\rm S}(x) + \sqrt{\mu} \sigma_{\rm SBI} p_{\rm SBI}(x) + (1-\sqrt{\mu}) \sigma_{\rm B} p_{\rm B}(x)}{ \mu \sigma_{\rm S} + \sqrt{\mu} \sigma_{\rm I} + \sigma_{\rm B}},
$$

where we have performed a small linear algebra maneuver, $\mu\sigma_{\rm S} + \sqrt{\mu} \sigma_{\rm I}+ \sigma_{\rm B} = (\mu-\sqrt{\mu})\sigma_{\rm S} + \sqrt{\mu}(\sigma_{\rm S} + \sigma_{\rm B} + \sigma_{\rm I}) + (1-\sqrt{\mu})\sigma_{\rm B} $ (multiplied with their probabilities) in the numerator.

We will judiciously choose the background-only hypothesis as our common denominator in order to estimate the following ratio:

$$
\frac{p_{\rm SBI} (x | \mu)}{p_{\rm B} (x)} = \frac{ (\mu - \sqrt{\mu}) \sigma_{\rm S} r_{\rm S}(x) + \sqrt{\mu} \sigma_{\rm SBI} r_{\rm SBI}(x) + (1-\sqrt{\mu}) \sigma_{\rm B} }{ \mu \sigma_{\rm S} + \sqrt{\mu} \sigma_{\rm I} + \sigma_{\rm B}},
$$

where $r_{\rm S,SBI}(x) \equiv p_{\rm S,SBI}(x)/p_{\rm B}$, and $r_{\rm B}(x) = 1$. In other words, performing DREs of two of the terms of the mixture model lets us perform DRE of the full SBI process for any value of $\mu$! This is what you will be performing in this tutorial to estimate confidence intervals on $\mu$ given an "observed" $gg \to 4\ell$ dataset.

*A little confusingly, our physics process (SBI) and statistical technique (NSBI) share similar acronyms. Always pay attention to the presence/lack of "N" to distinguish the two.