# Learning OpenAi-Gym Boxing

This is the project for generating the learning agent of Atari 'Boxing' game using reinforcement learning and OpenAi-Gym.

## Getting Started

### Environment

Pip modules

```
gym == 0.9.3

matplotlib == 2.0.2

numpy == 1.13.1

python == 2.7.13

scikit-image == 0.13.0

tensorflow-gpu == 1.0.1
```

This program might not be compatible with other versions of modules.

## Run the Program

If you want to train the agent, run the following command in the project directory:
```
$ python ./src/main.py --train --episodes <the_number_of_episodes> [--save <dir_to_save_result>]
```

If you want to test the agent with certain model, run the following command in the project directory:
```
$ python ./src/main.py --test --model <dir_where_model_is_saved> --episodes <the_number_of_episodes> [--save <dir_to_save_result>]
```

## Test the Program

If you are using Linux, run the following commands in the project directory:

```
$ ./tests/test_train.sh

$ ./tests/test_test.sh
```

## Authors

* **INKYU PARK** - *Initial work*
