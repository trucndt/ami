We include three sets of python files for the three datasets:

- `celeba-eval-noldp.py` and `celeba-exp-noldp.py`: CelebA
- `cifar-eval-noldp.py` and `cifar-exp-noldp.py`: CIFAR-10
- `imgnet-eval-noldp.py` and `img-exp-noldp.py`: ImageNet

The `*-exp*.py` is used to train the malicious parameters, while `*-eval*.py` is used to evaluate the parameters and calculate the attack success rate. Below we show an example on how to run the attack on CIFAR-10, other datasets follow the same syntax.

To run the attack, execute the following command:

```bash
$ python cifar-exp-noldp.py -r NUMNEURONS -o OUTPUT_PATH
```
- `NUMNEURONS` is the number of neurons in the first layer (parameter $r$)
- `OUTPUT_PATH` is the path to the output directory

Example:
```bash
$ python cifar-exp-noldp.py -r 2000 -o res/
```

To get the attack success rate, execute the following command:
```bash
$ python cifar-eval-noldp.py -r NUMNEURONS -o OUTPUT_PATH
```
- `OUTPUT_PATH` is the path to the output files of the attack

Example:
```bash
$ python cifar-eval-noldp.py -r 2000 -o res/
```
This will output TPR, TNR, and Adv


For full usage, run `python cifar-exp-noldp.py -h` and `python cifar-eval-noldp.py -h`