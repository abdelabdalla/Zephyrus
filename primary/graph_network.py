# This file is based on code developed by DeepMind Technologies Limited.
# The use of this code follows the rules laid out by Apache License, Version 2.0.
# Source: https://github.com/deepmind/deepmind-research/blob/master/learning_to_simulate/graph_network.py

import graph_nets as gn
import sonnet as snt
import tensorflow as tf


def build_mlp(
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int) -> snt.Module:
    return snt.nets.MLP(
        output_sizes=[hidden_size] * num_hidden_layers + [output_size])


class EncodeProcessDecode(snt.AbstractModule):

    def __init__(
            self,
            latent_size: int,
            mlp_hidden_size: int,
            mlp_num_hidden_layers: int,
            num_message_passing_steps: int,
            output_size: int,
            name: str = "EncodeProcessDecode"):

        super().__init__(name=name)

        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size

        with self._enter_variable_scope():
            self._networks_builder()

    def _build(self, input_graph: gn.graphs.GraphsTuple) -> tf.Tensor:

        latent_graph_0 = self._encode(input_graph)
        latent_graph_m = self._process(latent_graph_0)

        return self._decode(latent_graph_m)

    def _networks_builder(self):
        def build_mlp_with_layer_norm():
            mlp = build_mlp(
                hidden_size=self._mlp_hidden_size,
                num_hidden_layers=self._mlp_num_hidden_layers,
                output_size=self._latent_size)
            return snt.Sequential([mlp, snt.LayerNorm()])

        encoder_kwargs = dict(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm)
        self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

        self._processor_networks = []
        for _ in range(self._num_message_passing_steps):
            self._processor_networks.append(
                gn.modules.InteractionNetwork(
                    edge_model_fn=build_mlp_with_layer_norm,
                    node_model_fn=build_mlp_with_layer_norm))

        self._decoder_network = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size)

    def _encode(
            self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:

        latent_graph_0 = self._encoder_network(input_graph)
        return latent_graph_0

    def _process(self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:

        latent_graph_prev_k = latent_graph_0
        for process_network_k in self._processor_networks:
            latent_graph_k = self._process_step(
                process_network_k, latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k

        latent_graph_m = latent_graph_prev_k
        return latent_graph_m

    def _process_step(
            self, process_network_k: snt.Module,
            latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:

        latent_graph_k = process_network_k(latent_graph_prev_k)

        latent_graph_k = latent_graph_k.replace(
            nodes=latent_graph_k.nodes + latent_graph_prev_k.nodes,
            edges=latent_graph_k.edges + latent_graph_prev_k.edges)
        return latent_graph_k

    def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
        return self._decoder_network(latent_graph.nodes)
