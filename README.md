# Constrained and unconstrained Deep Image Prior optimization models with automatic regularization

### [Paper (Journal)](https://link.springer.com/article/10.1007/s10589-022-00392-w) | [Paper (ResearchGate)](https://www.researchgate.net/publication/362299603_Constrained_and_unconstrained_deep_image_prior_optimization_models_with_automatic_regularization)

# Abstract
Deep Image Prior (DIP) is currently among the most efficient unsupervised deep learning based methods for ill-posed inverse problems in imaging. This novel framework relies on the implicit regularization provided by representing images as the output of generative Convolutional Neural Network (CNN) architectures. So far, DIP has been shown to be an effective approach when combined with classical and novel regularizers. Unfortunately, to obtain appropriate solutions, all the models proposed up to now require an accurate estimate of the regularization parameter. To overcome this difficulty, we consider a locally adapted regularized unconstrained model whose local regularization parameters are automatically estimated for additively separable regularizers. Moreover, we propose a novel constrained formulation in analogy to Morozov’s discrepancy principle which enables the application of a broader range of regularizers. Both the unconstrained and the constrained models are solved via the proximal gradient descent-ascent method. Numerical results demonstrate the robustness with respect to image content, noise levels and hyperparameters of the proposed models on both denoising and deblurring of simulated as well as real natural and medical images.

# Citing
Please consider to cite CDIP-RED if you find it helpful.

```BibTex
@article{cascarano2022constrained,
  title={Constrained and unconstrained deep image prior optimization models with automatic regularization},
  author={Cascarano, Pasquale and Franchini, Giorgia and Kobler, Erich and Porta, Federica and Sebastiani, Andrea},
  journal={Computational Optimization and Applications},
  pages={1--25},
  year={2022},
  publisher={Springer}
}
 ```