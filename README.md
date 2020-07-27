# Python GPU Experiment Parallelization

[한국어](https://nyanye.com/gpu/2020/07/27/multi-gpu/)

Sometimes it is tough to implement your model to efficiently utilize multi-gpu environments using [data parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html) and [model parallelism](https://github.com/kakaobrain/torchgpipe). When your code is mixed with multiple versions of deeplearning framework's API such as TensorFlow v1 compatible apis and TensorFlow v2 compatible apis the code is really fragile and official parallelization practices doesn't simply work right away. For some deeplearning practices we eventually spend more time to implement it in parallel way.

The time is also an expensive resource, however we can't just let residual GPUs idle forever.

The concept [task parallelism](https://en.wikipedia.org/wiki/Task_parallelism) is also a great choice for gpu experiments and it is commonly favored for many reasons.

## Example

* When you work for company and the business-hour is limited,
    * You have to train 'model A' which takes about 3 days for convergence
    * You have no idea if only 1 experiment is enough
    * You have 4 gpus
    * Today is friday

If you train 'model A' in traditional data parallel style,
* Imagine 4 gpus can train your model in about 1 day if everything worked very well.
* If you stop training at designated duration or epochs, you can program to train different model variants when the first one is over. However you cannot check their progress at once.

If you train 'model A' and 3 more variant such as 'model A1', 'model A2', 'model A3',
* Imagine 4 gpus can train your 4 models in 3 days

...and if you're gonna check results at monday, you would able to compare 4 models at once when you chose to parallelize task (experiment)

### Different scenarios

Train with one gpu, validate and debug model or it's application powered by gpu
- GPU 0: Training
- GPU 1: Validation / Debugging

Train multiple models or code base simultaneously
- GPU 0: Model A
- GPU 1: Model B
- GPU 2: Model A with hparam B
- GPU 3: Model B with hparam A

## Use [GPUtil](https://github.com/anderskm/gputil) to automatically choose one best available gpu

### Install GPUtil

```bash
$ pip install gputil
```

### Automate choosing single GPU

./train.py

```python
import os
import GPUtil


def choose_one_gpu():
    """set CUDA_VISIBLE_DEVICES environment variable to utilize one best gpu """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUtil.getFirstAvailable()[0])


def train():
    choose_one_gpu()
    some_train_ops(...)


if __name__ == "__main__":
    train()
```

### Experiment Parallelization

```bash
# You can use terminal multiplexer to manage your training sessions
# CAUTION: DO NOT EXECUTE EVERYTHING AT ONCE,
# WAIT FOR THE APPLICATION TO ALLOCATE THEIR MEMORY AND START UTILIZING GPU
python model_a/train.py model_a/configs/experiment_01.yaml &
sleep 5
python model_a/train.py model_a/configs/experiment_02.yaml &
sleep 5
python model_b/train.py model_b/configs/experiment_01.yaml &
sleep 5
python model_b/train.py model_b/configs/experiment_02.yaml &
```
