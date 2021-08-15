# This file is based on code developed by DeepMind Technologies Limited.
# The use of this code follows the rules laid out by Apache License, Version 2.0.
# Source: https://github.com/deepmind/deepmind-research/blob/master/learning_to_simulate/noise_utils.py

import tensorflow as tf

from primary import ns_simulator


def get_random_walk_noise_for_velocity_sequence(
        velocity_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""

    acc_sequence = ns_simulator.time_diff(velocity_sequence)

    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random_walk:
    # std_last_step**2 = num_velocities * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    # TODO(alvarosg): Make sure this is consistent with the value and
    # description provided in the paper.
    num_acc = acc_sequence.shape.as_list()[1]
    acc_sequence_noise = tf.random.normal(
        tf.shape(acc_sequence),
        stddev=noise_std_last_step / num_acc ** 0.5,
        dtype=velocity_sequence.dtype)

    # Apply the random walk.
    velocity_sequence_noise = tf.cumsum(acc_sequence_noise, axis=1)

    # Integrate the noise in the velocity to the positions, assuming
    # an Euler intergrator and a dt = 1, and adding no noise to the very first
    # position (since that will only be used to calculate the first position
    # change).
    vel_sequence_noise = tf.concat([
        tf.zeros_like(acc_sequence_noise[:, 0:1]),
        tf.cumsum(acc_sequence_noise, axis=1)], axis=1)

    return vel_sequence_noise
