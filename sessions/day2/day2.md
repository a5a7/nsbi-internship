# Day 2: ML Overviews, Differentiable Programing and NSBI

## Neural simulation-based inference (NSBI)

In this tutorial, we will setting a confidence interval on a single parameter of interest using NSBI. It consists of individual "chapters" covering:

1. Data exploration: phenomenology of the $gg \to (h^{\ast} \to) ZZ \to 4\ell$ process.
2. Training classifiers that approximate ratios of the likelihood ratio (CARL).
3. Evaluating the accuracy of the density ratio estimates via diagnostics.
4. Using the CARL estimates to perform neural simulation-based inference of the Higgs signal strength.

### Density Ratio Esimation (DRE) via NSBI

The specific flavour of NSBI performed in the context of maximum likelihood fits is to estimate the ratio of probability density of an event between two hypotheses:

$$
r (x ; H_1, H_2) \equiv \frac{p(x | H_1)}{p(x | H_2)}.
$$

Henceforth, $H_1$ and $H_2$ are referred to as _numerator_ and _denominator_ hypotheses, respectively.

### Cross-sections and matrix elements, latent variables and observables

The set of features that represents an event will be denoted $x$. These are really physical _observables_ in a sense that they measurable in a real dataset, without any underlying knowledge/assumptions. 

We are interested in (differential) cross-sections of an event occuring via a physics process, referred to as _hypothesis_, as a function of observables, as this encodes the total rate and probability of the event under the hypothesis. From a theory (or _simulator_) perspective, these quantities are calculated involving "truth" and unobservable quantities  are referred to as _latent_ variables.

$$ \frac{d\sigma}{dx} = \sigma p(x) = \int dz \; p(x | z) |\mathcal{M}(z)|^2 $$

Note: in our specific physics problem (see below), the in-principle complicated relation between matrix-element squared and the differential cross-section from the convolution over the latent space is trivialized due to the following factors:

1. Disregard detector effects (fair approximation given excellent  resolution with respect to lepton measurements at LHC).
2. No parton showering effects for the final states (only considering LO events with no extra radiation).

This essentially boils down the in-general-intractable convolution into a identity mapping up to some PDF factors, i.e. $p(x | z) \to \delta(x - z) {\rm PDF(z)}$. These simplifications make the datasets & task ideal as a tutorial, but always keep in mind that a robust NSBI method is one that works even without these advantages modulo numerical convergence.

### Physics at hand: Off-shell Higgs production

The physics problem at hand to be studied is the off-shell production of the Higgs boson. Disregarding the $q\bar{q}$-initiated processes (imagine an LGC: the Large Gluon Collider), the leading-order $gg \to 4\ell$ diagrams are:

1. Higgs signal via gluon-fusion production, $gg \to h^{\ast} \to ZZ \to 4\ell$.
2. Continuum background via box diagram, $gg \to ZZ \to 4\ell$.

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
\frac{p_{\rm SBI} (x | \mu)}{p_{\rm B} (x)} = \frac{ (\mu - \sqrt{\mu}) \sigma_{\rm S} p_{\rm S}(x) + \sqrt{\mu} \sigma_{\rm SBI} p_{\rm SBI}(x) + (1-\sqrt{\mu}) \sigma_{\rm B} }{ \mu \sigma_{\rm S} + \sqrt{\mu} \sigma_{\rm I} + \sigma_{\rm B} },
$$

where we have performed a small linear algebra maneuver, $\mu\sigma_{\rm S} + \sqrt{\mu} \sigma_{\rm I}+ \sigma_{\rm B} = (\mu-\sqrt{\mu})\sigma_{\rm S} + \sqrt{\mu}(\sigma_{\rm S} + \sigma_{\rm B} + \sigma_{\rm I}) + (1-\sqrt{\mu})\sigma_{\rm B} $ (multiplied with their probabilities) in the numerator.

Note: A little confusingly, our physics process (SBI) and statistical technique (NSBI) share similar acronyms. Always pay attention to the presence/lack of "N" to distinguish the two.
