# visual_attention
Code and supplements for the article "Visual Attention Through Uncertainty Minimization in Recurrent Generative Models"


All relevant code can be found in the <src> directory. The model is specified in <model.py> and <modules.py>. The training and testing loops can be found in <trainer.py>. <data_loader.py> and <config.py> handle the data and configuration of all simulations. The configuration files used to train the models in the articles can be found in the <configs> folder.

<Reproduce Experiments.ipynb> calls the test runs with the pretrained models in the <ckpts> dir and displays the figures shown in the article. To run the notebook the MNIST, translated MNIST and, cluttered MNIST data have to be generated using https://github.com/deepmind/mnist-cluttered
 and placed into a <data/> folder.

All required python packages should be found in <requirements.txt>.