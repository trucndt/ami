# Active Membership Inference Attack under Local Differential Privacy in Federated Learning

## Dependencies
This codebase has been developed and tested only with python 3.8.10 and pytorch 1.7.0, on a linux 64-bit operation system.

### conda
We have prepared a file containing the same environment specifications that we use for this project. To reproduce this environment (only on a linux 64-bit OS), execute the following command:

```bash
$ conda create --name <name_env> --file spec-list.txt
```

- `name_env` is the name you want for your environment

Activate the created environment with the following command:

```bash
$ conda activate <name_env>
```


## Preprocessing

1. Follow the `README.md` files in these directories `data_celebA/celeba`, `data_imgnet/` to download the datasets
2. Run `$ python preprocessing.py`

## Usage

We include two directories:

- `noldp`: AMI attack without LDP
- `ldp`: AMI attack under LDP (including BitRand and OME)

Follow the `README.md` files in those directories

## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
