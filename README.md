# Learning Energy-Based Models by Diffusion Recovery Likelihood

I don't want the global variable FLAGS from `absl.flag`.

I create a ipynb notebook `difrec_tf2.ipynb` and verify it on colab. **Today is 2023.01.27**

You need to upload the related  `*.py` files to colab, and then

you can change the settings and note that 

1. the default model T6 with (6,30) is slow.
2. (1000, 0) will fail, so I don't know how to train T1k 


```
hps['num_res_blocks'] = 2
hps['num_diffusion_timesteps'] = 6
hps['mcmc_num_steps'] = 30
```

Just click the button or go to check the file (a colab button in it)

<a href="https://colab.research.google.com/github/sndnyang/DiffusionRecoveryLikelihood-ipynb/blob/master/difrec_tf2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>



## One more thing

https://github.com/sndnyang/DiffusionRecoveryLikelihood-PyTorch

The pytorch implementation (I hope it can run smoothly but there may exists bugs)

###  one more thing
Will upload T1k setting soon!   --- 2 years later

# Official Repo

https://github.com/ruiqigao/recovery_likelihood

If you find their work helpful to your research, please cite:
```
@article{gao2020learning,
  title={Learning Energy-Based Models by Diffusion Recovery Likelihood},
  author={Gao, Ruiqi and Song, Yang and Poole, Ben and Wu, Ying Nian and Kingma, Diederik P},
  journal={arXiv preprint arXiv:2012.08125},
  year={2020}
}
```