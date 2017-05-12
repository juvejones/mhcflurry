# Copyright (c) 2016. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import (
    print_function,
    division,
    absolute_import,
)

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Concatenate
from keras.layers import Input, regularizers
import theano

theano.config.exception_verbosity = 'high'


def make_network(
        input_size,
        embedding_input_dim=None,
        embedding_output_dim=None,
        layer_sizes=[100],
        activation="tanh",
        init="glorot_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=True,
        initial_embedding_weights=None,
        embedding_init_method="glorot_uniform",
        optimizer="rmsprop",
        model = None,
        rna_expression=True,
        ###loss="mse",
        loss="binary_crossentropy" ### Use binary classification objective
        ):
    if model == None:
        sequence_input = Input(shape=(input_size,))

    if embedding_input_dim:
        if not embedding_output_dim:
            raise ValueError(
                "Both embedding_input_dim and embedding_output_dim must be "
                "set")

        if initial_embedding_weights:
            n_rows, n_cols = initial_embedding_weights.shape
            if n_rows != embedding_input_dim or n_cols != embedding_output_dim:
                raise ValueError(
                    "Wrong shape for embedding: expected (%d, %d) but got "
                    "(%d, %d)" % (
                        embedding_input_dim, embedding_output_dim,
                        n_rows, n_cols))
            sequence_layer = Embedding(
                input_dim=embedding_input_dim,
                output_dim=embedding_output_dim,
                input_length=input_size,
                weights=[initial_embedding_weights],
                dropout=dropout_probability)(sequence_input)
        else:
            sequence_layer = Embedding(
                input_dim=embedding_input_dim,
                output_dim=embedding_output_dim,
                input_length=input_size,
                embeddings_initializer=embedding_init_method,
                dropout=dropout_probability)(sequence_input)
        
        sequence_layer = Flatten()(sequence_layer)
        #####################
        # add fixed feature #
        #####################
        add_input = Input(shape = (1,))
        add_layer = Dense(1, input_shape=(1,))(add_input)
        merged_layer = Concatenate()([sequence_layer, add_layer])
 
        input_size = input_size * embedding_output_dim

    layer_sizes = (input_size,) + tuple(layer_sizes)

    for i, dim in enumerate(layer_sizes):
        if i == 0:
            # input is only conceptually a layer of the network,
            # don't need to actually do anything
            continue

        previous_dim = layer_sizes[i - 1]

        # hidden layer fully connected layer
        hidden_layer1 = Dense(
                input_dim=previous_dim,
                output_dim=dim,
                init=init, activation = activation)(merged_layer)
        #model.add(Activation(activation))

        #if rna_expression:
            #hidden_layer2 = Dense(dim, input_shape=(1,), 
            #    activation = activation, 
            #    kernel_regularizer=regularizers.l1(0.01))(add_layer)
        #   hidden_layer1 = Concatenate()([hidden_layer1, add_layer])
        
        if batch_normalization:
            hidden_layer = BatchNormalization()(hidden_layer1)

        if dropout_probability > 0:
            hidden_layer = Dropout(dropout_probability)(hidden_layer1)

    # output
    output_layer = Dense(
        input_dim=layer_sizes[-1],
        output_dim=1,
        init=init, activation = output_activation)(hidden_layer)
    #model.add(Activation(output_activation))
    model = Model(inputs=[sequence_input, add_input], outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def make_hotshot_network(
        peptide_length=9,
        n_amino_acids=20,
        **kwargs):
    """
    Construct a feed-forward neural network whose inputs are binary vectors
    representing a "one-hot" or "hot-shot" encoding of a fixed length amino
    acid sequence.
    """
    return make_network(input_size=peptide_length * n_amino_acids, **kwargs)


def make_embedding_network(
        peptide_length=9,
        n_amino_acids=20,
        embedding_output_dim=20,
        **kwargs):
    """
    Construct a feed-forward neural network whose inputs are vectors of integer
    indices.
    """
    return make_network(
        input_size=peptide_length,
        embedding_input_dim=n_amino_acids,
        embedding_output_dim=embedding_output_dim,
        **kwargs)
