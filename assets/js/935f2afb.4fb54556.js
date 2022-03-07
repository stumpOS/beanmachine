"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[53],{1109:function(e){e.exports=JSON.parse('{"pluginId":"default","version":"current","label":"Next","banner":null,"badge":false,"className":"docs-version-current","isLast":true,"docsSidebars":{"someSidebar":[{"type":"category","collapsed":true,"collapsible":true,"label":"Overview","items":[{"type":"link","label":"Why Bean Machine?","href":"/docs/overview/why_bean_machine","docId":"overview/why_bean_machine/why_bean_machine"},{"type":"link","label":"Quick Start","href":"/docs/overview/quick_start","docId":"overview/quick_start/quick_start"},{"type":"link","label":"Modeling","href":"/docs/overview/modeling","docId":"overview/modeling/modeling"},{"type":"link","label":"Inference","href":"/docs/overview/inference","docId":"overview/inference/inference"},{"type":"link","label":"Analysis","href":"/docs/overview/analysis","docId":"overview/analysis/analysis"},{"type":"link","label":"Installation","href":"/docs/overview/installation/","docId":"overview/installation/installation"}]},{"type":"category","collapsed":true,"collapsible":true,"label":"Framework","items":[{"type":"link","label":"Worlds and Variables","href":"/docs/world","docId":"framework_topics/world"},{"type":"category","collapsed":true,"collapsible":true,"label":"Inference Methods","items":[{"type":"link","label":"Overview","href":"/docs/inference","docId":"framework_topics/inference/inference"},{"type":"link","label":"Single-Site Ancestral MH","href":"/docs/ancestral_metropolis_hastings","docId":"framework_topics/inference/ancestral_metropolis_hastings"},{"type":"link","label":"Single-Site Random Walk MH","href":"/docs/random_walk","docId":"framework_topics/inference/random_walk"},{"type":"link","label":"Single-Site Uniform MH","href":"/docs/uniform_metropolis_hastings","docId":"framework_topics/inference/uniform_metropolis_hastings"},{"type":"link","label":"Hamiltonian Monte Carlo","href":"/docs/hamiltonian_monte_carlo","docId":"framework_topics/inference/hamiltonian_monte_carlo"},{"type":"link","label":"No-U-Turn Sampler","href":"/docs/no_u_turn_sampler","docId":"framework_topics/inference/no_u_turn_sampler"},{"type":"link","label":"Newtonian Monte Carlo","href":"/docs/newtonian_monte_carlo","docId":"framework_topics/inference/newtonian_monte_carlo"}]},{"type":"category","collapsed":true,"collapsible":true,"label":"Custom Inference","items":[{"type":"link","label":"Overview","href":"/docs/programmable_inference","docId":"framework_topics/custom_inference/programmable_inference"},{"type":"link","label":"Custom Proposers","href":"/docs/custom_proposers","docId":"framework_topics/custom_inference/custom_proposers"},{"type":"link","label":"Compositional Inference","href":"/docs/compositional_inference","docId":"framework_topics/custom_inference/compositional_inference"},{"type":"link","label":"Block Inference","href":"/docs/block_inference","docId":"framework_topics/custom_inference/block_inference"}]},{"type":"category","collapsed":true,"collapsible":true,"label":"Model Evaluation","items":[{"type":"link","label":"Diagnostics","href":"/docs/diagnostics","docId":"framework_topics/model_evaluation/diagnostics"}]},{"type":"category","collapsed":true,"collapsible":true,"label":"Development","items":[{"type":"link","label":"Logging","href":"/docs/logging","docId":"framework_topics/development/logging"}]}]},{"type":"category","collapsed":true,"collapsible":true,"label":"Advanced","items":[{"type":"link","label":"Bean Machine Graph Inference","href":"/docs/beanstalk","docId":"overview/beanstalk/beanstalk"}]},{"type":"category","collapsed":true,"collapsible":true,"label":"Tutorials","items":[{"type":"link","label":"Tutorials","href":"/docs/tutorials","docId":"overview/tutorials/tutorials"},{"type":"link","label":"Coin flipping","href":"/docs/overview/tutorials/Coin_flipping/CoinFlipping","docId":"overview/tutorials/Coin_flipping/CoinFlipping"},{"type":"link","label":"Hierarchical regression","href":"/docs/overview/tutorials/Hierarchical_regression/HierarchicalRegression","docId":"overview/tutorials/Hierarchical_regression/HierarchicalRegression"}]}]},"docs":{"contributing":{"id":"contributing","title":"Contributing Docs","description":"This document describes how to add and update markdown content that is presented in this Docusaurus 2 project."},"framework_topics/custom_inference/block_inference":{"id":"framework_topics/custom_inference/block_inference","title":"Block Inference","description":"Single-site inference in Bean Machine is a powerful abstraction that allows the inference engine to separately sample values for random variables in your model. While efficient in sampling high-dimensional models, single-site inference may not be suitable for models with highly correlated random variables. This is where Bean Machine\'s CompositionalInference API becomes handy: it allows us to \\"block\\" multiple nodes together and make proposals for them jointly.","sidebar":"someSidebar"},"framework_topics/custom_inference/compositional_inference":{"id":"framework_topics/custom_inference/compositional_inference","title":"Compositional Inference","description":"Sometimes it might be hard to pick a single algorithm that performs well on the entire model. For example, while gradient-based algorithms such as No-U-Turn Sampler and Newtonian Monte Carlo generally yield high number of effective samples, they can only handle random variables with continuous support. On the other hand, while single site algorithms make it easier to update high-dimensional models by proposing only one node at a time, they might have trouble updating models with highly correlated random variables. Fortunately, Bean Machine supports composable inference through the CompositionalInference class, which allows us to use different inference methods to update different subset of nodes and to \\"block\\" multiple nodes together so that they are accepted/rejected jointly by a single Metropolis-Hastings step. In this doc, we will cover the basics of CompositionalInference and how to mix-and-match different inference algorithms. To learn about how to do \\"block inference\\" with CompositionalInference, see Block Inference.","sidebar":"someSidebar"},"framework_topics/custom_inference/custom_proposers":{"id":"framework_topics/custom_inference/custom_proposers","title":"Custom Proposers","description":"API","sidebar":"someSidebar"},"framework_topics/custom_inference/programmable_inference":{"id":"framework_topics/custom_inference/programmable_inference","title":"Programmable Inference","description":"Programmable inference is a key feature of Bean Machine, and is achieved through three key techniques:","sidebar":"someSidebar"},"framework_topics/custom_inference/transforms":{"id":"framework_topics/custom_inference/transforms","title":"Transforms","description":"Bean Machine provides flexibility for users to specify transformations on a per-variable basis. This gives Bean Machine powerful functionality."},"framework_topics/development/logging":{"id":"framework_topics/development/logging","title":"Logging","description":"Logging in Bean Machine is provided through the logging module in Python. It is recommended that users get familiar with the basics (logger, handler, levels, etc.) of this module before reading further.","sidebar":"someSidebar"},"framework_topics/inference/ancestral_metropolis_hastings":{"id":"framework_topics/inference/ancestral_metropolis_hastings","title":"Single-Site Ancestral Metropolis-Hastings","description":"Ancestral Metropolis-Hastings is one of the most fundamental Bayesian inference methods. In ancestral Metropolis-Hastings, values are sampled from the model\'s priors, and samples are accepted or rejected based on the sample\'s Metropolis acceptance probability. As such, ancestral Metropolis-Hastings is a very general inference method, making no strong assumptions about the structure of the model. However, this generality may lead it to be rather inefficient for many models.","sidebar":"someSidebar"},"framework_topics/inference/hamiltonian_monte_carlo":{"id":"framework_topics/inference/hamiltonian_monte_carlo","title":"Hamiltonian Monte Carlo","description":"Hamiltonian Monte Carlo (HMC) is a sampling algorithm for differentiable random variables which uses Hamiltonian dynamics. By randomly drawing a momentum for the kinetic energy and treating the true posterior as the potential energy, HMC is able to simulate trajectories which explore the space. Intuitively, this can be viewed as starting with a marble at a point inside a bowl, flicking the marble in a random direction, and then following the marble as it rolls around. The position of the marble represents the sample, the flick represents the momentum, and the shape of the bowl in combination with the force of gravity represents our true posterior.","sidebar":"someSidebar"},"framework_topics/inference/inference":{"id":"framework_topics/inference/inference","title":"Inference Methods","description":"Posterior distributions can often only be estimated, as the solutions to such problems in general have no closed-form. Bean Machine\'s inference methods include sequential sampling techniques known as Markov chain Monte Carlo (MCMC) to generate samples representative of this distribution. These posterior distribution samples are the main output of Bean Machine: with enough samples, they will asymptotically converge to the true posterior.","sidebar":"someSidebar"},"framework_topics/inference/newtonian_monte_carlo":{"id":"framework_topics/inference/newtonian_monte_carlo","title":"Newtonian Monte Carlo","description":"Newtonian Monte Carlo (NMC) (Arora, et al., 2020) is a second-order gradient-based Markov chain Monte Carlo (MCMC) algorithm that uses the first- and second-order gradients to propose a new value for a random variable.","sidebar":"someSidebar"},"framework_topics/inference/no_u_turn_sampler":{"id":"framework_topics/inference/no_u_turn_sampler","title":"No-U-Turn Sampler","description":"The No-U-Turn Sampler (NUTS) (Hoffman and Gelman, 2014) algorithm is an inference algorithm for differentiable random variables which uses Hamiltonian dynamics. It is an extension to the Hamiltonian Monte Carlo (HMC) inference algorithm.","sidebar":"someSidebar"},"framework_topics/inference/random_walk":{"id":"framework_topics/inference/random_walk","title":"Single-Site Random Walk Metropolis-Hastings","description":"Random Walk Metropolis-Hastings is a simple, minimal MCMC inference method. Random Walk Metropolis-Hastings is single-site by default, following the philosophy of most inference methods in Bean Machine, and accordingly multi-site inference patterns are well supported. Random Walk Metropolis-Hastings follows the standard Metropolis-Hastings algorithm of sampling a value from a proposal distribution, and then running accept-reject according to the computed ratio of the proposed value. This is further detailed in the docs for Ancestral Metropolis-Hastings. This tutorial describes the proposal mechanism, describes adaptive Random Walk Metropolis-Hastings, and documents the API for the Random Walk Metropolis-Hastings algorithm.","sidebar":"someSidebar"},"framework_topics/inference/uniform_metropolis_hastings":{"id":"framework_topics/inference/uniform_metropolis_hastings","title":"Single-Site Uniform Metropolis-Hastings","description":"Single-Site Uniform Metropolis-Hastings is used to infer over variables that have discrete support, for example random variables with Bernoulli and Categorical distributions. It is overall very similar to Ancestral Metropolis-Hastings. However, it is designed so that it will even explore discrete samples that are unlikely under the prior distribution.","sidebar":"someSidebar"},"framework_topics/model_evaluation/diagnostics":{"id":"framework_topics/model_evaluation/diagnostics","title":"Diagnostics","description":"Intra- and inter-chain diagnostics can tell us how well a particular inference algorithm performed on the model. Two common diagnostics","sidebar":"someSidebar"},"framework_topics/model_evaluation/model_comparison":{"id":"framework_topics/model_evaluation/model_comparison","title":"Model Comparison","description":"Let\'s suppose we have a problem for which we have several candidate models. How do we determine which is best? We can run several diagnostics on the posterior samples, but those will only reveal how well inference has converged on each specific model, without testing whether that is indeed the best model. To compare different models, we can instead assess performance directly on held-out test data."},"framework_topics/model_evaluation/posterior_predictive_checks":{"id":"framework_topics/model_evaluation/posterior_predictive_checks","title":"Posterior Predictive Checks","description":"Evaluating probabilistic models is a challenging effort and an open research problem.  A common way to evaluate how well your model fits the data is to do posterior predictive checking , i.e. how well a model\'s predictions match the data."},"framework_topics/world":{"id":"framework_topics/world","title":"Worlds and Variables","description":"Worlds","sidebar":"someSidebar"},"mdx":{"id":"mdx","title":"Powered by MDX","description":"You can write JSX and use React components within your Markdown thanks to MDX."},"overview/analysis/analysis":{"id":"overview/analysis/analysis","title":"Analysis","description":"Inference results are useful not only for learning posterior distributions, but for verifying that inference ran correctly. We\'ll cover common techniques for analyzing results in this section. As is the case for everything else in this Overview, the code for this section is available as a notebook on GitHub and Colab.","sidebar":"someSidebar"},"overview/api/api":{"id":"overview/api/api","title":"Application Programming Interface (API)","description":"Bean Machine"},"overview/beanstalk/beanstalk":{"id":"overview/beanstalk/beanstalk","title":"Bean Machine Graph Inference","description":"What is Bean Machine Graph Inference?","sidebar":"someSidebar"},"overview/inference/inference":{"id":"overview/inference/inference","title":"Inference","description":"Inference is the process of combining a model with data to obtain insights, in the form of probability distributions over values of interest.","sidebar":"someSidebar"},"overview/installation/installation":{"id":"overview/installation/installation","title":"Installation","description":"Did You Check Out Colab?","sidebar":"someSidebar"},"overview/modeling/modeling":{"id":"overview/modeling/modeling","title":"Modeling","description":"Declarative Style","sidebar":"someSidebar"},"overview/packages/packages":{"id":"overview/packages/packages","title":"Hierarchical Mixed Effects","description":"Packages in Bean Machine let a user reuse tested, proven code for specific purposes, relieving a user from needing to write their own custom Bean Machine logic."},"overview/quick_start/quick_start":{"id":"overview/quick_start/quick_start","title":"Quick Start","description":"Let\'s quickly translate the model we discussed in \\"Why Bean Machine?\\" into Bean Machine code! Although this will get you up-and-running, it\'s important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine.","sidebar":"someSidebar"},"overview/tutorials/Coin_flipping/CoinFlipping":{"id":"overview/tutorials/Coin_flipping/CoinFlipping","title":"Coin flipping","description":"<LinkButtons","sidebar":"someSidebar"},"overview/tutorials/Hierarchical_regression/HierarchicalRegression":{"id":"overview/tutorials/Hierarchical_regression/HierarchicalRegression","title":"Hierarchical regression","description":"<LinkButtons","sidebar":"someSidebar"},"overview/tutorials/tutorials":{"id":"overview/tutorials/tutorials","title":"Tutorials","description":"These Bean Machine tutorials demonstrate various types of statistical models that users can build in Bean Machine. Play around with them in an interactive Google Colab notebook!","sidebar":"someSidebar"},"overview/why_bean_machine/why_bean_machine":{"id":"overview/why_bean_machine/why_bean_machine","title":"Why Bean Machine?","description":"Bean Machine is a probabilistic programming language that makes developing and deploying generative probabilistic models intuitive and efficient. This page describes the motivation for using Probabilistic Programming in general, and Bean Machine advantages specifically.","sidebar":"someSidebar"},"tutorials/listing":{"id":"tutorials/listing","title":"listing","description":"Bean Machine tutorials are contained in these Notebooks. The page you are currently viewing should not be displayed on the website; it exists to point you to our internal Bento Notebooks, which should be used instead. Please find the tutorial links listed in the \\"Tutorials\\" section of our table of contents, included below."}}}')}}]);