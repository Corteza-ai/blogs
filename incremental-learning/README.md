# Incremental learning
### This code is a companion of the post: https://corteza.ai/news-blog-incremental-learning/

Clone this repository to your local machine:

```bash
git clone https://github.com/Corteza-ai/blogs.git
```

Download the data from [here](https://www.kaggle.com/datasets/sanikamal/17-category-flower-dataset), and unzip it in your working directory.

To avoid conflicts, you must then configure a Python virtual environment.
Please follow the next steps to get started:

### 1. cd to the corresponding folder.
### 2. Create a Python virtual environment

* Windows users:

```bash
py -m venv myenv
```

* For Linux/Mac users:

```bash
python3 -m venv myenv
```

### 3. Activate the virtual environment.

* Windows users:

```bash
.\myenv\Scripts\activate
```

* For Linux/Mac users:

```bash
source myenv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 4.1 Instal pytorch

<details>
<summary>Windows users</summary>
CPU only

```bash
pip install torch torchvision
```
GPU

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
```
</details>

<details>
<summary>Linux users</summary>
CPU only

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```
GPU

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
```
</details>

<details>
<summary>MacOS users</summary>
CPU only

```bash
pip install torch torchvision
```
</details>

## Usage

To get a plot with a random sample of images from the dataset execute the following command:

```bash
python save_sample_images.py
```

To train the model in an end-to-end fashion, please use the command below:

```bash
python main.py --mode end-to-end --epochs 500 --lr 0.005
```

To train with the incremental approach, run the code as follows:

```bash
python main.py --model incremental --epochs 20 --last-num-epochs 50 --lr 0.005
```

where _epochs_ denotes the number of epochs to train each layer at each step in the incremental approach, and the _last-num-epochs_ flag is used to define the number of epochs to fine-tune the model.

To compare the performance concerning the image reconstruction of the trained models, run the following command:

```bash
python comparison.py
```

The above saves a _png_ file into your working directory with the original images (at the top), the end-to-end model reconstructions (in the middle), and the greedy layer-wise model reconstructions (at the bottom).

If you want to replicate the experiments described in the original post, use the following command:

```bash
python main.py --mode blog-experiments
```

You can use our pre-trained models, as shown in the following example:
```python
import torch
autoencoder = torch.load('pre-trained-models/glw_trained_model.pt')

inputs = torch.rand(10, 3, 224, 224)
inputs = inputs.to('cuda')
with torch.no_grad():
    outputs = model(inputs)
```
The model is loaded to the GPU by default. Set the ```map_location``` parameter of the ```torch.load``` method to 'cpu' if you only want to use cpu.

```python
autoencoder = torch.load('pre-trained-models/glw_trained_model.pt', map_location='cpu')
```