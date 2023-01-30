# Auto-Encoders
### This code is related to the following post: https://corteza.ai/news-blog-autoencoders-part-ii-implementation/

## Preliminaries

Clone this repository to your local machine:


```bash
git clone https://github.com/Corteza-ai/auto-encoder-post.git
```

To avoid conflicts, you must then configure a Python virtual environment.
Follow the next steps to get started:

### 1. cd to the source code folder.
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

You can interact with this [Jupyter notebook](./Auto-Encoders.ipynb), type and run the following:


```bash
jupyter notebook
```
and click to the corresponding file to start the notebook. 

Alternatively, you can use the following command to train and save a PyTorch model of a CAE: 

```bash
python main.py --data_path ./17_flowers/train --epochs 500 --lr 0.01
```

## Acknowledgements

The data set used was obtained from: ```Nilsback, M. E., & Zisserman, A. (2006, June). A visual vocabulary for flower classification. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06) (Vol. 2, pp. 1447-1454). IEEE.```
