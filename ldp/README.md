We include three sets of python files for the three datasets:

- `celeba-eval.py` and `celeba-exp.py`: CelebA
- `cifar-eval.py` and `cifar-exp.py`: CIFAR-10
- `imgnet-eval.py` and `img-exp.py`: ImageNet

The `*-exp.py` is used to train the malicious parameters, while `*-eval.py` is used to evaluate the parameters and calculate the attack success rate. Below we show an example on how to run the attack on CIFAR-10, other datasets follow the same syntax.

*Note that the code uses multiprocessing*

To run the attack, execute the following command:

```bash
$ python cifar-exp.py --eps EPS -m MECH -o OUTPUT_PATH
```
- `EPS` is the epsilon value
- `MECH` is either `BitRand` or `OME`
- `OUTPUT_PATH` is the path to the output directory

Example:
```bash
$ python cifar-exp.py --eps 5 -m BitRand -o res/ 
```

To get the attack success rate, execute the following command:
```bash
$ python cifar-eval.py --eps EPS -m MECH -o OUTPUT_PATH
```
- `OUTPUT_PATH` is the path to the output files of the attack

Example:
```bash
$ python cifar-eval.py --eps 5 -m BitRand -o res/
```
This will output TPR, TNR, and Adv


For full usage, run `python cifar-exp.py -h` and `python cifar-eval.py -h`