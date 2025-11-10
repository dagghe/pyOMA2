---
title: 'pyOMA2: A Python module for conducting operational modal analysis'
tags:
  - Python
  - operational modal analysis
  - dynamics of structures
  - system identification
  - ambient vibrations
authors:
  - name: Dag P. Pasca
    orcid: 0000-0002-3830-2835
    corresponding: true
    affiliation: "1"
  - name: Diego Federico Margoni
    affiliation: 2
affiliations:
 - name: Norsk Treteknisk Institutt, Oslo, Norway
   index: 1
 - name: Politecnico di Torino, Italy
   index: 2

date: 12 September 2024
bibliography: paper.bib
---

## Summary

Operational modal analysis (OMA) has garnered considerable attention from
the engineering community in recent years and has established itself as
the preferred method for estimating the modal properties of structures in
structural health monitoring (SHM) applications, particularly in civil engineering.
The key advantage of OMA over experimental modal analysis (EMA) is its
ability to derive modal parameters solely from output measurements taken
during the structure's normal operation. This makes OMA a more practical
and efficient approach, as opposed to the traditional EMA, which requires
both input and output data.

## Statement of need

`pyOMA2` is the latest and improved version of the `pyOMA` module [@pasca2022pyoma], a Python
library specifically designed for conducting operational modal analysis.
While its predecessor relied on procedural workflows, `pyOMA2`
fully utilises Python's object-oriented capabilities to offer a
comprehensive suite of tools for performing OMA.

Notable improvements over the previous version include support for single- and multi-setup measurements, allowing users to handle multiple acquisitions that combine reference and roving sensors; enhanced user-friendliness through a broad range of tools for pre-processing and visualising data; interactive plotting that enables users to select desired modes directly from algorithm-generated graphs; a geometry-definition feature to visualise mode shapes on tested structures; and, since version 1.1.1, the possibility to estimate uncertainty bounds of modal properties for the SSI family of algorithms.

The following algorithms are included in the module:

- Frequency domain decomposition (FDD) [@brincker2001modal];
- Enhanced frequency domain decomposition (EFDD) [@brincker2001damping];
- Frequency spatial domain decomposition (FSDD) [@zhang2010frequency];
- Reference-based covariance driven stochastic subspace identification (SSIcov) [@van2012subspace;@peeters1999reference;@reynders2012system];
- Reference-based data driven stochastic subspace identification (SSIdat) [@van2012subspace;@peeters1999reference;@reynders2012system];
- Poly-reference least square frequency domain (pLSCF) [@peeters2004polymax];

The multi-setup analyzes can be performed according the so-called post separate estimation
re-scaling (PoSER) approach as well as with the so-called pre-global estimation re-scaling (PreGER)
approach [@brincker2015introduction;@rainieri2014operational;@dohler2013efficient;@amador2021robust]. The calculation of the uncertainty bounds for the SSI family of algorithms follows the efficient implementation by Döhler and colleagues [@dohler2011subspace;@dohler2013efficient;@dohler2013uncertainty]. The interested reader may refer to the extensive scientific literature on the subject for further information.

A few commercial software programs implements the algorithms mentioned above.
The most well-known presumably are ARTeMIS [@solutions2001artemis], by Structural
Vibration Solutions, and MACEC, a Matlab toolbox for modal testing and
OMA [@reynders2014macec]. When it comes to open source modules the only ones
available to the authors best knowledge are the first version of `pyOMA`
[@pasca2022pyoma] and `Koma` [@koma], which is also an open-source Python library
available on GitHub. It provides tools for OMA, focusing on simplicity and ease of use.
`Koma` is designed to be a lightweight alternative to more general libraries like `pyOMA`,
making it suitable for smaller projects.

The module's reliability and applicability for research purposes have been
demonstrated by the authors through various studies
[@alaggio2021two;@aloisio2020dynamic;@simoncelli2023intensity].
Additionally, the module has gained traction within the research community,
as evidenced by its use in studies by
@saharan2023convolutional, @croce2023towards, @talebi2023interoperability, @abuodeh2023examining, and others.

## Module's structure

The module is structured into three primary levels:

1. At the first level are the `setup` classes. Users instantiate these classes by providing a data array and the sampling frequency for a single setup scenario, or a list of data arrays and their respective sampling frequencies, and reference indices, for a multi-setup scenario.
2. The second level comprises the `algorithms` classes. Users can instantiate the algorithms they wish to run and then add them to the setup class.
3. The third level contains the `support` classes, which serve as auxiliary components to the first two levels. This level includes various specialized classes:
    - `result` classes, where outcomes are stored.
    - `geometry` classes, for storing geometric data.
    - `run_param` classes, where parameters used for running the algorithms are kept.
    - Dedicated classes for animating mode shapes and interacting with plots generated by the algorithm classes.

In addition to the levels depicted in the figure, there is a further
level not shown, comprised of the set of functions internally called
by the class methods. Many of these functions represent an updated
version of those available in our previous release, `pyOMA`.

![Schematic organisation of the module showing inheritance between classes](../docs/img/info.pdf)

## Documentation

A comprehensive documentation for `pyOMA2`, including examples, is available at
[https://pyoma.readthedocs.io/en/main/](https://pyoma.readthedocs.io/en/main/).

## Acknowledgements

We acknowledge contributions from Angelo Aloisio and Marco Martino Rosso.

## References
