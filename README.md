## Introduce

This repo is the released code for "Downstream-Pretext Domain
Knowledge Traceback for Active Learning" in TMM.

In the code/reproduce folder, we reproduce the results by our model and save the selection results (2%, 4%, 6%, 8%, 10%, 12%, 14%, 16%, 18%, 20% labeled samples). You can evaluate the performances of these selections by execute the acc.py and other files. This program will train a ResNet-18 with our labeled samples and evaluate on the test set.



### Prerequisites:
- Linux 
- Python 3.5/3.6
- NVIDIA GPU + CUDA CuDNN
- torch>=1.1.0
- torchvision>=0.3.0
- numpy>=1.16.4
- scikit-learn>=0.21.1
- scipy>=1.2.1
- tqdm >= 4.31.1

### Evaluate our reproduced result
You can evaluate our reproduced results by running the command under './TMM'.  (This command should be runned under the directory code/reproduce/  )


```
python3 acc100.py 0
```
The accuracy on cifar100 will be printed on the screen.

```
python3 acc100lt.py 0
```
The accuracy on long-tailed cifar100 will be printed on the screen.

```
python3 acc10lt.py 0
```
The accuracy on long-tailed cifar10 will be printed on the screen.

```
python3 acc10.py 0
```
The accuracy on cifar10 will be printed on the screen.


### Draw the results
You can draw the results by run the python scripts under './draw_cvpr23'.  (This command should be runned under the directory code/reproduce/  )

