# This file is based on code developed by DeepMind Technologies Limited.
# The use of this code follows the rules laid out by Apache License, Version 2.0.
# Source: https://github.com/deepmind/deepmind-research/blob/master/learning_to_simulate/learned_simulator.py

import graph_nets as gn
import sonnet as snt
import tensorflow as tf

from primary import connectivity_utils
from primary import graph_network


class NSSimulator(snt.AbstractModule):

    def __init__(
            self,
            graph_network_kwargs,
            name='NSSimulator'):
        super().__init__(name=name)

        self.graph_network_kwargs = graph_network_kwargs
        with self._enter_variable_scope():
            self._graph_network = graph_network.EncodeProcessDecode(
                output_size=2, **graph_network_kwargs)

    def _build(self, velocity_sequence, n_nodes, n_conn, node_locations, node_connections):
        input_graphs_tuple = self._encoder_preprocessor(
            velocity_sequence, n_nodes, n_conn, node_locations, node_connections)

        acceleration = self._graph_network(input_graphs_tuple)

        next_vel = self._decoder_postprocessor(acceleration, velocity_sequence)

        return next_vel

    def _encoder_preprocessor(self, velocity_sequence, n_nodes, n_conn, node_locations, node_connections):

        (senders, receivers, n_edge) = connectivity_utils.get_connectivity_for_batch_pyfunc(
            node_locations, node_connections, n_nodes, n_conn)

        node_features = []

        flat_velocity_sequence = snt.MergeDims(start=1, size=2)(velocity_sequence)
        node_features.append(flat_velocity_sequence)

        edge_features = []

        send = tf.gather(node_locations, senders)
        rec = tf.gather(node_locations, receivers)

        relative_displacements = send - rec
        edge_features.append(relative_displacements)
        relative_distances = tf.norm(relative_displacements, axis=-1, keepdims=True)
        edge_features.append(relative_distances)

        return gn.graphs.GraphsTuple(
            nodes=tf.concat(node_features, axis=-1),
            edges=tf.concat(edge_features, axis=-1),
            globals=None,  # self._graph_net will appending this to nodes.
            n_node=n_nodes,
            n_edge=n_edge,
            senders=senders,
            receivers=receivers,
        )

    def _decoder_postprocessor(self, acceleration, velocity_sequence):

        most_recent_velocity = velocity_sequence[:, -1]

        new_velocity = most_recent_velocity + acceleration

        return new_velocity

    def get_predicted_and_target_normalized_accelerations(
            self, next_velocity, n_nodes, n_conn, velocity_sequence, node_locations, node_connections,
            velocity_sequence_noise):
        noisy_velocity_sequence = velocity_sequence + velocity_sequence_noise
        input_graphs_tuple = self._encoder_preprocessor(noisy_velocity_sequence, n_nodes, n_conn, node_locations,
                                                        node_connections)
        predicted_acceleration = self._graph_network(input_graphs_tuple)

        next_velocity_adjusted = next_velocity + velocity_sequence_noise[:, -1]
        target_acceleration = self._inverse_decoder_postprocessor(next_velocity_adjusted, noisy_velocity_sequence)

        return predicted_acceleration, target_acceleration

    def _inverse_decoder_postprocessor(self, next_velocity, velocity_sequence):
        previous_velocity = velocity_sequence[:, -1]
        acceleration = next_velocity - previous_velocity

        return acceleration


def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]
