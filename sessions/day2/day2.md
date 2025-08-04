# Day 2: ML Overviews, Differentiable Programing and NSBI

## Neural simulation-based inference (NSBI)

NSBI refers to a class of machine learning techniques used to perform inference when one has:

$$
x \sim p(x | H),
$$

where $x$ is the set of features that specifies an event, i.e. _observables_ of a dataset. 
The NSBI performed in the context of maximum likelihood fits is to estimate the ratio of probability density (DRE) of an event between two hypotheses:

$$
r (x ; H_1, H_2) \equiv \frac{p(x | H_1)}{p(x | H_2)}.
$$

Henceforth, $H_1$ and $H_2$ are referred to as _numerator_ and _denominator_ hypotheses, respectively. As to *why* one might want to estimate these ratios, you should recall the Neyman-Pearson Lemma and associated foundations of frequentist statistics.

In this tutorial, we will be implementing NSBI to measure a single paramter of interest.
The tutorial consists of individual "chapters" covering:

1. Data exploration: phenomenology of the $gg \to (h^{\ast} \to) ZZ \to 4\ell$ process.
2. Training classifiers that approximate ratios of the likelihood ratio (CARL).
3. Evaluating the accuracy of the density ratio estimates via diagnostics.
4. Using the CARL estimates to perform neural simulation-based inference of the Higgs signal strength.

### Cross-sections and matrix-elements; observables and latent variables

The process by which the likelihood is rendered intractable can be expressed as a convolution over the latent space to the observable space; in the HEP context, the matrix-element squared of a partonic process in a $pp$ collision must undergo various non-perturbative and sampling processes accounting for PDFs, parton showering and hadronization, and detector acceptance/efficiency/resolution effects. These processes are modelled using MC methods

$$ \frac{d\sigma}{dx} = \sigma p(x) = \int dz \; p(x | z) |\mathcal{M}(z)|^2 $$

Note: in our specific physics problem (see below), this complicated evolution from latent-to-observable space is actually trivialized due to the following factors:

1. Disregard detector effects (fair approximation given excellent  resolution with respect to lepton measurements at LHC), and
2. No parton showering effects for the final states (only considering LO events with no extra radiation),

which essentially reduce the convolution into a identity mapping modulo some PDF factors, i.e. $p(x | z) \to \delta(x - z) {\rm PDF(z)}$. These simplifications make the datasets & task ideal as a tutorial, but always keep in mind that NSBI method works in general with these effects, as long as the NNs are trained sufficiently.

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
\frac{p_{\rm SBI} (x | \mu)}{p_{\rm B} (x)} = \frac{ (\mu - \sqrt{\mu}) \sigma_{\rm S} r_{\rm S}(x) + \sqrt{\mu} \sigma_{\rm SBI} r_{\rm SBI}(x) + (1-\sqrt{\mu}) \sigma_{\rm B} }{ \mu \sigma_{\rm S} + \sqrt{\mu} \sigma_{\rm I} + \sigma_{\rm B},
$$

where $r_{\rm S,SBI}(x) \equiv p_{\rm S,SBI}(x)/p_{\rm B}$, and $r_{\rm B}(x) = 1$. In other words, performing DREs of two of the terms of the mixture model lets us perform DRE of the full SBI process for any value of $\mu$! This is what you will be performing in this tutorial to estimate confidence intervals on $\mu$ given an "observed" $gg \to 4\ell$ dataset.

*A little confusingly, our physics process (SBI) and statistical technique (NSBI) share similar acronyms. Always pay attention to the presence/lack of "N" to distinguish the two.
