Learning Atari Boxing with OpenAi-Gym
#####################################

This is the project for generating the learning agent of Atari ‘Boxing’
game using reinforcement learning and OpenAi-Gym.


.. contents::

.. section-numbering::


Getting Started
===============

Environment Setup
-----------------

To run the 'setup.py' code, you need below programs:

.. code-block::
    pip~=9.0.1

    python~=2.7

Next, run following command in the project directory to resolve
dependencies:

.. code-block:: bash
    $ python setup.py install


Run the Program
===============

If you want to train the agent, run the following command in the project
directory:

.. code-block:: bash
    $ python ./src/main.py --train --episodes <the_number_of_episodes> [--save <dir_to_save_result>]

If you want to test the agent with certain model, run the following
command in the project directory:

.. code-block:: bash
    $ python ./src/main.py --test --model <dir_where_model_is_saved> --episodes <the_number_of_episodes> [--save <dir_to_save_result>]


Test the Program
================

Test whether the program is running
-----------------------------------

If you are using Linux, run the following commands in the project
directory:

.. code-block:: bash
    # Debian, Ubuntu, etc.
    $ ./tests/test_train.sh

    $ ./tests/test_test.sh

Unit test
---------

Currently we do not provide a unit test.


Authors
=======

-  **INKYU PARK** - *Initial work*