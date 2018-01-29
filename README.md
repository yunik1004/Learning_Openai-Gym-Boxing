# Learning Atari Boxing with OpenAi-Gym

This is the project for generating the learning agent of Atari 'Boxing' game using reinforcement learning and OpenAi-Gym.

## Getting Started

### Environment Setup

To run this program, you need below programs:

```
cmake>=3.5(only if you want to rebuild the module)

pip~=9.0.1

python~=2.7
```

(If you want to install the module, you don't need to do next step)

Next, run following command in the project directory to resolve dependencies:

```bash
$ pip install -r requirements.txt
```

(If you don't want to install the module, you don't need to do following steps)

If you want to rebuild the module, run following command in the project directory:

```bash
$ cmake CmakeLists.txt
```

Run following command in the project directory to install the module:

```bash
$ make
$ make install
```

## Run the Program

If you want to train the agent, run the following command in the project directory:

```bash
$ python ./src/main.py --train --episodes <the_number_of_episodes> [--save <dir_to_save_result>]
```

If you want to test the agent with certain model, run the following command in the project directory:

```bash
$ python ./src/main.py --test --model <dir_where_model_is_saved> --episodes <the_number_of_episodes> [--save <dir_to_save_result>]
```

## Test the Program

### Test whether the program can run

Run the following commands in the project directory:

```bash
#Linux
$ ./tests/test_train.sh

$ ./tests/test_test.sh
```

### Unit test

Currently we do not provide a unit test.

## Authors

* **INKYU PARK** - *Initial work*
