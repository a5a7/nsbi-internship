# Day 2: ML Overviews, Differentiable Programing and NSBI

## Neural simulation-based inference (NSBI)

In this tutorial, we will setting a confidence interval on a single parameter of interest using NSBI. It consists of individual "chapters" covering:

1. Data exploration: phenomenology of the $gg \to (h^{\ast} \to) ZZ \to 4\ell$ process.
2. Training classifiers that approximate ratios of the likelihood ratio (CARL).
3. Evaluating the accuracy of the density ratio estimates via diagnostics.
4. Using the CARL estimates to perform neural simulation-based inference of the Higgs signal strength.

### Physics at hand: Off-shell Higgs production

The physics problem at hand to be studied is the off-shell production of the Higgs boson via gluon-fusion and subsequent decay to $ZZ \to 4\ell$ system.

### Cross-sections and matrix elements, latent variables and observables

The most complete set of features that represents an event in terms of experimental observables, which will be denoted $x$ , is the four-momenta of each of the four leptons, ordered by their $p_{\rm T}$.

$$ x \equiv (E_\ell, {\bf p}_{\ell}),\, \ell = 1,2,3,4$$

Events corresponding to each hypothesis and their lepton kinematics have been prepared for you to open. Additionally, associated with each event is also a real-valued *weight* that corresponds to the value of the differential cross-section of the event:

$$ \frac{d\sigma}{dx} = \sigma p(x) = \int dz \; p(x | z) |\mathcal{M}(z)|^2 $$

The underlying matrix-element is in principle a function of parton-level momenta configuration, i.e. latent (not observable) variables. In our specific case of the 4-lepton channel, the relation is simplified due to the following factors:

1. Disregard detector effects (fair approximation given excellent  resolution with respect to lepton measurements at LHC).
2. No parton showering effects for the final states (only considering LO events with no extra radiation).

This essentially boils down the in-general-intractable convolution into a identity mapping up to some PDF factors, i.e. $p(x | z) \to \delta(x - z) {\rm PDF(z)}$. These simplifications a