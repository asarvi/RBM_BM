This repository contains implementations of two neural network models: a Fully Connected Boltzmann Machine (FCBM) and a Restricted Boltzmann Machine (RBM). 
## Implementation Details for Boltzmann Machine (BM)

To run the code in this repository, you will need the latest version of the following libraries:

•  `torch`

•  `torch.optim`

•  `torch.nn`

•  `torch.nn.functional`

•  `numpy`

•  `torchvision`

•  `math`

•  `matplotlib`

Here an overview of how the code for the Boltzmann machine operates is provided. The code itself contains some useful comments and explanations.

The dataset used in this projects is MNIST, accessed via torch.utils.data, which has been divided into three parts: training, testing, and validation(10% of training set). To make training efficient, the batch size is set to 50.
Using this code the dataset will be downloaded and ready to use: 
```python
import torchvision
from torch.utils.data import DataLoader, Subset

#init batch size
batch_size = 50

#define transform
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,
                download=True,transform=transform
)

# define validation set and split dataset
validation_split = 0.1
validation_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - validation_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, 
                                    [train_size, validation_size])

# create data loaders for training, validation, and test sets
train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
test_dataset = torchvision.datasets.MNIST( root='./data',train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(test_dataset,batch_size,shuffle=False)
validation_loader = DataLoader(validation_dataset,batch_size,shuffle=False)



```

The FCBM class is defined with 784 visible (input) units, according to the 28x28 pixel size of MNIST images, and 612 hidden units. The number of hidden units was selected according to the methods provided in one of the reference papers.

### Key Functions and Model Architecture

•  sample_BM(p): Samples binary states based on probabilities p.

•  forward(v): Conducts the forward pass, implementing the Contrastive Divergence algorithm.

•  free_energy(v): Calculates the free energy for a given visible state v, essential for loss computation during training, where the free energy is defined as:

$$ F(v) = -\sum_{i} a_i v_i - \sum_{j} \log(1 + e^{(b_j + \sum_{i} v_i W_{ij})}) $$

Training is conducted over 10 epochs, a balance between efficiency and performance. The learning rate was fine-tuned to 0.08 after experimentation, as it shows acceptable performance according to output(loss) graphs. For optimizer I used SGD.

During the training, the model applies forward passes, loss calculations, backpropagation, and optimization, and after each epoch, results are printed as outputs, which can be seen after each cell runs in the code.
### Experimentation and Results

To evaluate the models, images of training samples and their corresponding generated samples are displayed. It was observed that the model more accurately generates digits composed of straight lines compared to those with curves, such as '2' and '8'.

## Implementation Details for Restricted Boltzmann Machine (RBM)

### Modifications to the Model Architecture
In the implementation of the RBM, several modifications were made to the architecture previously described. The RBM is not a fully connected model. Therefore, certain functions within the `RBM` class have been defined.

### Key Functions in the RBM Class
•  `vis_to_hid`: This function computes the probability distribution of the hidden units and generates a sample of hidden units.

•  `hid_to_vis`: This function calculates the probability distribution of the visible units and produces a sample of visible units.

•  `forward`: Within the `forward` method, the `vis_to_hid` and `hid_to_vis` functions are used to obtain the final output sample.


### Experimentation and Results
The code was run using various batch sizes, learning rates, and epoch counts. The selected parameters  were based on their performance, based on the output graphs.
Finally, batch size is set to 80, learning rate is 0.06, number of epochs is 15 and number of hidden layers is set to 300 after testing various numbers.
The quality of generated digits was acceptable and the samples were similar to training set. However, the model seems to be generating some noises in digits.
These samples can also be seen at the end of each code file.


