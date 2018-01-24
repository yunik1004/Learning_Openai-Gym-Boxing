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

tensorflow-gpu == 1.0.1
```

This program might not be compatible with other versions of modules.

## Running the Program

If you want to train the agent, run the following command:
```
$ python main.py --train [--save dir_to_save_result]
```

If you want to test the agent with certain model, run the following command:
```
$ python main.py --test --model dir_where_model_is_saved [--save dir_to_save_result]
```

## Authors

* **INKYU PARK** - *Initial work*
