# -*- coding: utf-8 -*-
# Author: Roman Haag

# TODO: This code should undergo major refactoring

import dace
from dace.memlet import Memlet, EmptyMemlet
from dace import SDFG, SDFGState
from dace.graph.nodes import Tasklet, NestedSDFG

import numpy as np
from collections import OrderedDict
import re

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Cannot use Tensorflow frontend without Tensorflow, "
        + "please install: https://www.tensorflow.org/install/"
    )

from tensorflow.python.framework import tensor_util


# http://stackoverflow.com/q/3844948/
def _checkEqualIvo(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def _tensortype(tensor: tf.Tensor):
    """ Returns a numpy type from a given TF tensor. """

    # Heuristics to determine op type
    if isinstance(tensor, tf.Operation):
        if len(tensor.outputs) == 1:
            tensor = tensor.outputs[0]
        elif len(tensor.inputs) == 1:
            tensor = tensor.inputs[0]
        elif _checkEqualIvo([inp.dtype for inp in tensor.inputs]):
            tensor = tensor.inputs[0]
        else:
            try:
                dtype = tensor.get_attr("T")
                if dtype.as_numpy_dtype == object:
                    raise NotImplementedError(
                        "Type %s is not a valid numpy type" % str(dtype)
                    )
                return dtype.as_numpy_dtype
            except ValueError:
                pass
            raise TypeError("Ambiguous type for operation %s" % tensor)

    if tensor.dtype.as_numpy_dtype == object:
        raise NotImplementedError(
            "Type %s is not a valid numpy type" % str(tensor.dtype)
        )

    if tensor.dtype.is_bool:
        return np.int32

    return tensor.dtype.as_numpy_dtype


def _tensorshape(tensor: tf.Tensor):
    if tensor.shape.dims is None or tensor.shape.dims == []:
        return 1  # Scalar
    return tensor.shape


def _string_builder(string):
    """ To match DaCe variable naming conventions, replaces all undesired 
        characters with "_".
    """
    newstring = string
    if string[0].isdigit():
        newstring = "_" + string
    out = re.sub("[^a-zA-Z0-9_]", "_", newstring)
    return out


def _name(tensor_or_op):
    if isinstance(tensor_or_op, tf.Operation):
        return None
    return _string_builder(tensor_or_op.name)


_LASTSESSION = 0


def _atomic_counter_generator():
    ctr = 0
    while True:
        ctr += 1
        yield ctr


_atomic_count = _atomic_counter_generator()


class TFSession:
    def __init__(self, name: str = "tfsession", seed: int = None, config=None):
        """ Creates a DaCe Tensorflow session.
            @param name: (optional) The name of the resulting SDFG.
            @param seed: (optional) Fix random seed.
        """
        self._internal_session = tf.Session(config=config)

        # Set for bookkeeping of already visited nodes
        self.visitedNodes = set()

        # Reinit state only used in training mode
        self.reinitState = None

        # Different input dictionaries
        self.constDict = dict()
        self.varDict = dict()
        self.inpDict = dict()
        self.reinitDict = dict()
        self.initDict = dict()

        self.training = False
        self.iterations = 1
        self.seed = seed
        self.graph = SDFG(name)
        self.kill = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def train(
        self,
        optimizer,
        initializer,
        iterations,
        feed_dict,
        nodes=None,
        output_gradients=False,
    ):
        """ Trains a subgraph for the specified number of iterations and 
            returns requested nodes after training.
            
            @param optimizer: A TensorFlow tf.Optimizer node.
            @param initializer: Either a list of global and local initializers
                                or one initializer.
            @param iterations: Number of training steps.
            @param feed_dict: Dictionary representing input values and arrays 
                              to feed in to the evaluator.
            @param nodes: (optional) A TensorFlow node or an iterable 
                          (e.g. list) of nodes to evaluate.
            @param output_gradients: A boolean, if set, will output all the gradients passed as the
                                     optimizer arument. This will assume optimizer contains the
                                     list of gradient tensors that will be added to the outputs.
            @return: A 2-tuple of (varDict, values) - the first is a dictionary
                     of all variables used in the network in arbitrary order,
                     and the second is a tuple of values in the same order as
                     `nodes`.
        """

        # Initialize a new SDFG
        self.graph = SDFG(self.graph.name)
        self.graph.propagate = False
        self.state = SDFGState("s0", self.graph)
        self.graph.add_node(self.state)
        self.iterations = iterations
        state = self.state
        sdfg = self.graph
        outputs = []
        output_names = []
        # init state
        s0 = state
        # computational state"
        s1 = sdfg.add_state("s1")
        # emtpy exit state
        s2 = sdfg.add_state("s2")
        # As currently output arrays of conflict resolution do not automaticly
        # get reinitialized in each state iterations, we have to manually do
        # it in this state.
        reinitState = sdfg.add_state("reinitialization")
        self.reinitState = reinitState
        # set training mode

        self.training = True

        # add edges between states
        sdfg.add_edge(
            s0, s1, dace.graph.edges.InterstateEdge(assignments=dict(__dacet1=0))
        )
        sdfg.add_edge(
            s1,
            reinitState,
            dace.graph.edges.InterstateEdge(
                condition=dace.properties.CodeProperty.from_string(
                    "__dacet1 <" + str(iterations - 1), dace.types.Language.Python
                ),
                assignments={"__dacet1": "__dacet1+1"},
            ),
        )
        sdfg.add_edge(reinitState, s1, dace.graph.edges.InterstateEdge())
        sdfg.add_edge(
            s1,
            s2,
            dace.graph.edges.InterstateEdge(
                condition=dace.properties.CodeProperty.from_string(
                    "__dacet1 >= " + str(iterations - 1), dace.types.Language.Python
                )
            ),
        )

        try:
            iter(initializer)
            initializer = list(initializer)
        except TypeError:
            initializer = [initializer]

        try:
            iter(nodes)
            nodes = list(nodes)
        except TypeError:
            nodes = [nodes]

        try:
            iter(optimizer)
            optimizer = list(optimizer)
        except TypeError:
            optimizer = [optimizer]

        ###########################
        # Prepare subgraph to process
        # If only one node was given, construct a list from it
        if not nodes == [None]:
            ops = [
                node if isinstance(node, tf.Operation) else node.op for node in nodes
            ]
            output_names = [
                _string_builder(node.name)
                if not isinstance(node, tf.Operation)
                else None
                for node in nodes
            ]

        # Visit initializer and create subgraph for init state
        # If only one node was given, construct a list from it

        init = [i if isinstance(i, tf.Operation) else i.op for i in initializer]
        self.visit_backwards(init)

        # Visit the rest of the nodes
        self.state = s1
        state = s1
        # As we are in a new state, all variable nodes should be revisited
        self.visitedNodes.clear()
        if not nodes == [None]:
            self.visit_backwards(ops)
        optimizer = [
            opt if isinstance(opt, tf.Operation) else opt.op for opt in optimizer
        ]
        self.visit_backwards(optimizer)
        ############################

        # Remove orphan nodes and register node types
        node_types = {}
        for state in self.graph.nodes():
            for node in state.nodes():
                if state.in_degree(node) + state.out_degree(node) == 0:
                    state.remove_node(node)
                    if node.label in self.constDict:
                        del self.constDict[node.label]
                elif isinstance(node, dace.graph.nodes.AccessNode):
                    node_types[node.data] = node.desc(self.graph).dtype.type
        ############################
        # Set up arguments
        sdfg_args = {}
        sdfg_args.update(self.constDict)
        sdfg_args.update(self.varDict)
        sdfg_args.update(self.inpDict)
        sdfg_args.update(self.reinitDict)
        sdfg_args.update(self.initDict)

        sdfg_args.update(
            {
                (k if isinstance(k, str) else _string_builder(k.name + "_Inp")): v
                for k, v in feed_dict.items()
            }
        )

        # Set scalar arguments to appropriate arrays of size 1
        sdfg_args.update(
            {
                k: (
                    v if isinstance(v, np.ndarray) else np.array(v, dtype=node_types[k])
                )
                for k, v in sdfg_args.items()
            }
        )

        ############################
        # Create output numpy arrays
        if output_gradients:
            for opt in optimizer:
                if isinstance(opt, tf.Tensor):
                    nodes.append(opt)
                    output_names.append(opt.name)
        if not nodes == [None]:
            outputs = {
                name: np.zeros(_tensorshape(node), dtype=_tensortype(node))
                for node, name in zip(nodes, output_names)
                if name is not None and name not in sdfg_args
            }
            outputs.update({k: v for k, v in sdfg_args.items() if k in output_names})

            sdfg_args.update(outputs)

        ############################
        # Mark outputs as non-transients
        for output in outputs:
            self.graph.arrays[output].transient = False
        ############################

        # Compile and call the SDFG
        self.graph.draw_to_file()
        compiled_sdfg = self.graph.compile(optimizer=False)
        compiled_sdfg(**sdfg_args)
        ############################

        # Return the outputs and weights

        return (
            self.varDict,
            tuple(
                outputs[output] if output is not None else None
                for output in output_names
            ),
        )

    def compile(self, nodes, name=None):
        """ Compiles a subgraph into a callable function, which is equivalent 
            to calling `run()`. 
            @param nodes: Node or an iterable (e.g. list) of nodes to evaluate.
            @param name: Name of the SDFG to create, or None for a unique name.
            @return: A function that receives a feed_dict, evaluates the nodes,
                     and returns a tuple of values in the same order as nodes.
        """
        # Create a unique name for this session
        if name is None:
            global _LASTSESSION
            _LASTSESSION += 1
            name = "tfsession%d" % _LASTSESSION

        # Initialize a new SDFG
        self.graph = SDFG(name)
        self.graph.propagate = False
        self.state = SDFGState("s0", self.graph)
        self.graph.add_node(self.state)
        self.visitedNodes.clear()
        ############################

        # Prepare subgraph to process
        total_nodes = []

        # Determine output type
        output_type = None
        if not isinstance(nodes, (list, tuple, dict)):  # iter() works in TensorFlow
            output_type = object
            total_nodes.append(nodes)
            output_names = _name(nodes)
        elif isinstance(nodes, dict):
            output_type = type(nodes)
            output_names = {}
            for k, node in nodes.items():
                try:
                    iter(node)
                    if isinstance(node, dict):
                        raise TypeError("Dictionaries of dictionaries unsupported")
                    total_nodes.extend(node)
                    output_names[k] = type(node)(_name(n) for n in node)
                except TypeError:
                    total_nodes.append(node)
                    output_names[k] = _name(node)
        elif isinstance(nodes, (list, tuple)):
            output_type = type(nodes)
            total_nodes.extend(nodes)
            output_names = output_type(_name(node) for node in nodes)
        else:
            raise TypeError("Unsupported type for fetches: " + str(type(nodes)))

        ops = [
            node if isinstance(node, tf.Operation) else node.op for node in total_nodes
        ]
        total_output_names = [
            _string_builder(node.name) if not isinstance(node, tf.Operation) else None
            for node in total_nodes
        ]

        self.kill = False
        self.visit_backwards(ops)
        if self.kill:
            raise NotImplementedError("Nodes listed above are not implemented")
        ############################

        # Remove orphan nodes and register node types
        node_types = {}
        for state in self.graph.nodes():
            for node in state.nodes():
                if state.in_degree(node) + state.out_degree(node) == 0:
                    state.remove_node(node)
                    if node.label in self.constDict:
                        del self.constDict[node.label]
                elif isinstance(node, dace.graph.nodes.AccessNode):
                    node_types[node.data] = node.desc(self.graph).dtype.type
        ############################

        # Set up arguments
        sdfg_args = {}
        sdfg_args.update(self.constDict)
        sdfg_args.update(self.varDict)
        sdfg_args.update(self.inpDict)
        sdfg_args.update(self.initDict)

        # Set scalar arguments to appropriate arrays of size 1
        sdfg_args.update(
            {
                k: (
                    v if isinstance(v, np.ndarray) else np.array(v, dtype=node_types[k])
                )
                for k, v in sdfg_args.items()
            }
        )
        ############################

        # Create output numpy arrays
        outputs = {
            name: np.zeros(_tensorshape(node), dtype=_tensortype(node))
            for node, name in zip(total_nodes, total_output_names)
            if name is not None and name not in sdfg_args
        }
        outputs.update({k: v for k, v in sdfg_args.items() if k in total_output_names})

        sdfg_args.update(outputs)

        ############################
        # Mark outputs as non-transients
        for output in outputs:
            self.graph.arrays[output].transient = False
        ############################

        # Compile the SDFG
        self.graph.fill_scope_connectors()
        self.graph.draw_to_file()
        compiled_sdfg = self.graph.compile(optimizer=False)

        ############################
        # Create the function that invokes the SDFG
        def call_func(feed_dict={}):
            invoke_args = dict(
                sdfg_args,
                **{
                    (k if isinstance(k, str) else _string_builder(k.name)): v
                    for k, v in feed_dict.items()
                }
            )

            compiled_sdfg(**invoke_args)

            # Single output
            if output_type is object:
                return outputs[output_names] if output_names is not None else None
            # Dictionary of lists/single outputs
            elif output_type is dict:
                out_dict = {}
                for k, v in output_names.items():
                    if isinstance(v, (list, tuple)):
                        out_dict[k] = type(v)(
                            outputs[vname] if vname is not None else None for vname in v
                        )
                    else:
                        out_dict[k] = outputs[v] if v is not None else None
                return out_dict
            # List of outputs
            else:
                return output_type(
                    outputs[output] if output is not None else None
                    for output in output_names
                )

        # Return the function
        return call_func

    def run(self, nodes, feed_dict={}, name=None):
        """ Evaluates a subgraph and returns a tuple of the evaluated nodes
            (behaves similarly to sess.run).
            @param nodes: Node or an iterable (e.g. list) of nodes to evaluate.
            @param feed_dict: Dictionary representing input values and arrays 
                              to feed in to the evaluator.
            @param name: Name of the SDFG to create, or None for a unique name.
            
            @return: Tuple or dictionary of values in the same order as `nodes`.
        """
        callfunc = self.compile(nodes, name=name)
        return callfunc(feed_dict=feed_dict)

    def dfs_nodes(self, source):
        """ Produce nodes in a depth-first-search (DFS) on a TensorFlow graph.
            @param source: The source node to start from.
            @return: A generator of nodes in the depth-first-search.       
            @note: Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
                    by D. Eppstein, July 2004.
        """

        # If source is a list of nodes (or any iterable), start from all
        try:
            iter(source)
            nodes = list(source)
        except TypeError:
            nodes = [source]

        visited = set()

        for start in nodes:
            if start in visited:
                continue
            visited.add(start)
            yield start

            inputSet = [inp.op for inp in start.inputs]
            inputSet.extend(list(start.control_inputs))
            stack = [(start, iter(inputSet))]
            while stack:
                parent, children = stack[-1]
                try:
                    child = next(children)

                    if child not in visited:
                        yield child
                        visited.add(child)

                        inputSet = [inp.op for inp in child.inputs]
                        inputSet.extend(list(child.control_inputs))
                        stack.append((child, iter(inputSet)))
                except StopIteration:
                    stack.pop()

    def visit_backwards(self, node):
        """ Visit a graph from an output node backwards to the inputs. """
        for node in self.dfs_nodes(node):
            if node not in self.visitedNodes:
                self.visit(node)

    def visit(self, node):
        """ Visit a specific node in the graph, creating the SDFG. """
        try:
            func = getattr(self, "visit_" + node.type)
        except AttributeError:
            # Only stop processing after all node types have been visited,
            # so that we know which implementations are missing.
            self.kill = True
            print("MISSING IMPLEMENTATION:", node.type)
        if self.kill == False:
            func(node)
        # mark node as visited
        self.visitedNodes.add(node)

    ######################################################################
    # Operator (TensorFlow graph node) visitors

    def visit_Add(self, node):
        self.visit_element_wise_op(node, "+")

    def visit_Mul(self, node):
        self.visit_element_wise_op(node, "*")

    def visit_Sub(self, node):
        self.visit_element_wise_op(node, "-")

    def visit_RealDiv(self, node):
        self.visit_element_wise_op(node, "/")

    def visit_Equal(self, node):
        self.visit_element_wise_op(node, "==")

    def visit_Const(self, node):
        state = self.state
        label = _string_builder(node.name + "_0")

        # Create DaCe shape
        shape = dace.properties.ShapeProperty.from_string(
            str(_tensorshape(node.outputs[0]))
        )
        # Create np array from tensor value
        npArray = tensor_util.MakeNdarray(node.get_attr("value")).reshape(shape)

        # Add to constDict so that it can be fed to the program
        self.constDict[label] = npArray.astype(_tensortype(node))

        nodeArray = list(filter(lambda a: a.label == label, self.state.nodes()))

        # If node already present set it non transient, otherwise add node
        if not nodeArray:
            dtype = dace.typeclass(_tensortype(node))
            state.add_array(label, shape, dtype, toplevel=True)
        else:
            nodeArray[0].desc(self.graph).transient = False

    def visit_NoOp(self, node):
        # no op case where nothing happens
        pass

    def visit_Pack(self, node):
        # we do nothing with this op
        pass

    def visit_StridedSlice(self, node):
        # we do nothing with this op
        pass

    def visit_VariableV2(self, node):

        state = self.state
        label = _string_builder(node.name) + "_0"
        shape = dace.properties.ShapeProperty.from_string(
            str(_tensorshape(node.outputs[0]))
        )

        try:
            outputNode = state.find_node(label)
            outputNode.desc(self.graph).transient = False
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node))
            state.add_array(label, shape, dtype)

        # If not already added to the varDict, add a placeholder
        # zero-initialized array to it so a value error is not triggered.
        if label not in self.varDict.keys():
            npArray = np.zeros(shape=shape)
            self.varDict[label] = npArray.astype(_tensortype(node))

    def visit_Assign(self, node):
        # Simple memcopy from input1 to input0 as assign has no outputlist but
        # input0 is the variable we want to assign
        # Modified to rely on only the second argument tensor for shape and
        # dtype.
        state = self.state
        label = _string_builder(node.inputs[1].name)
        try:
            fillNode = state.find_node(label)
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.inputs[1]))
            shape = dace.properties.ShapeProperty.from_string(
                str(_tensorshape(node.inputs[1]))
            )
            fillNode = state.add_transient(
                name=label, shape=shape, dtype=dtype, toplevel=True
            )

        label = _string_builder(node.inputs[0].name)
        try:
            emptyNode = state.find_node(_string_builder(node.inputs[0].name))
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.inputs[1]))
            shape = dace.properties.ShapeProperty.from_string(
                str(_tensorshape(node.inputs[1]))
            )
            assert dtype is not None
            assert shape is not None
            emptyNode = state.add_transient(
                name=label, shape=shape, dtype=dtype, toplevel=True
            )
        dims = self.get_default_dims(node.inputs[1])
        memlet = Memlet.simple(emptyNode, ",".join(dims))
        state.add_edge(fillNode, None, emptyNode, None, memlet)

    def visit_AssignVariableOp(self, node):
        self.visit_Assign(node)

    def visit_Placeholder(self, node):

        outputShape = []
        outputParams = []
        outputDims = []
        inputShape = []
        inputParams = []
        inputDims = []
        outputTensor = node.outputs[0]
        state = self.state
        label = _string_builder(node.name + "_0")

        # Check if the node is already in the graph and get as a list
        try:
            outputNode = state.find_node(label)

        except (LookupError):
            outputNode = self.create_and_add_output_node(node)

        dtype = _tensortype(node)

        # If we are in training mode, we set up another map to reduce the huge
        # (iterations x batchsize x size of input) input to one dimension less
        if self.training:
            # Output dimensions of the map

            outputDims = self.get_default_dims(outputTensor)
            outputParams = self.get_default_params(outputTensor, 1)
            outputShape = list(map(str, _tensorshape(outputTensor)))

            # Prepend the iterations dimension to the input (t1=iterations)
            inputShape.append(str(self.iterations))
            inputShape.extend(outputShape)
            inputParams.append("i0")
            inputParams.extend(outputParams)
            inputDims.append("__dacet1:__dacet1+1")
            inputDims.extend(outputDims)

            # create node for the training examples
            shape = dace.properties.ShapeProperty.from_string(",".join(inputShape))
            dtype = _tensortype(node)
            inputNode = state.add_array(
                name=label + "_Inp", shape=shape, dtype=dace.typeclass(dtype)
            )

            # create and add mapp
            mapDict = dict(zip(inputParams, inputDims))
            inMemletDict = dict(j0=Memlet.simple(inputNode, ",".join(inputParams)))
            outMemletDict = dict(out=Memlet.simple(outputNode, ",".join(outputParams)))
            code = "out = j0"
            tasklet, map_entry, map_exit = state.add_mapped_tasklet(
                label, mapDict, inMemletDict, code, outMemletDict
            )
            state.add_edge(
                inputNode,
                None,
                map_entry,
                None,
                Memlet.simple(inputNode, ",".join(inputDims)),
            )
            state.add_edge(
                map_exit,
                None,
                outputNode,
                None,
                Memlet.simple(outputNode, ",".join(outputDims)),
            )

            # If training example node is not already in inputDict, add a
            # zero array. This prevents DaCe from raising a key error when
            # trying to call the dace function if we only execute a subgraph
            # where it does not appear. This might not be necessary any longer.
            if label + "_Inp" not in self.inpDict.keys():
                self.inpDict[label + "_Inp"] = np.zeros(
                    tuple(map(int, (inputShape))), dtype=dtype
                )

            # If we are not training, set the output non transient and add to
            # input dict
        else:
            outputNode.desc(self.graph).transient = False
            self.inpDict[label] = np.zeros(
                tuple(map(int, (outputNode.desc(self.graph).shape))), dtype=dtype
            )

    def visit_TruncatedNormal(self, node):
        # Creates a truncated normal array and adds it to initDict
        state = self.state
        label = _string_builder(node.name + "_0")
        # Check if already in graph, set non-transient. Otherwise add to graph.
        try:
            outputNode = state.find_node(label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        seed = 0 if self.seed is None else self.seed

        array = tf.truncated_normal(node.outputs[0].shape, seed=seed).eval(
            session=self._internal_session
        )
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_RandomStandardNormal(self, node):

        state = self.state
        label = _string_builder(node.name + "_0")

        try:
            outputNode = state.find_node(label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        array = tf.random_normal(node.outputs[0].shape, seed=self.seed).eval(
            session=self._internal_session
        )
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_RandomUniform(self, node):
        # Creates a random uniform array and adds it to initDict
        state = self.state
        label = _string_builder(node.name + "_0")
        # Check if already in graph, set non-transient. Otherwise add to graph.
        try:
            outputNode = state.find_node(label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        seed = 0 if self.seed is None else self.seed

        array = tf.random_uniform(node.outputs[0].shape, seed=seed).eval(
            session=self._internal_session
        )
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_RandomUniformInt(self, node):
        # Creates a random uniform array and adds it to initDict
        state = self.state
        label = _string_builder(node.name + "_0")
        # Check if already in graph, set non-transient. Otherwise add to graph.
        try:
            outputNode = state.find_node(label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        seed = 0 if self.seed is None else self.seed

        array = tf.random_uniform(
            node.outputs[0].shape,
            dtype=tf.as_dtype(_tensortype(node)),
            minval=node.inputs[1],
            maxval=node.inputs[2],
            seed=seed,
        ).eval(session=self._internal_session)
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_Fill(self, node):
        # Fills an array with a scalar input value
        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            # Scalar input is at position 1
            if count == 1:
                inp, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inp.desc(self.graph))
                inputNodes.append(inp)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)

        for out in node.outputs:
            params = self.get_default_params(out, 1)
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0] + outputParams[0]
        mapRange = inputDims[0] + outputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Slice(self, node):
        begin_positions = self._internal_session.run(node.inputs[1])
        sizes = self._internal_session.run(node.inputs[2])
        end_positions = begin_positions + sizes
        inputNode, _, _ = self.create_and_add_input_node(node.inputs[0])
        outputNode = self.create_and_add_output_node(node)[0]
        input_subset = [
            str(b) + ":" + str(e) for b, e in zip(begin_positions, end_positions)
        ]
        sliceMemlet = Memlet.simple(
            inputNode,
            ",".join(input_subset),
            other_subset_str=",".join(self.get_default_dims(node.outputs[0])),
        )
        self.state.add_edge(inputNode, None, outputNode, None, sliceMemlet)

    def visit_Mean(self, node):
        outputNode = self.create_and_add_output_node(node)[0]
        outputDims = self.get_default_dims(node.outputs[0])
        inputNode, params, dims = self.create_and_add_input_node(node.inputs[0])
        reduction_axes = self._internal_session.run(node.inputs[1])
        reduction_axes.sort()
        norm = 1
        for i in reduction_axes:
            norm *= inputNode.desc(self.graph).shape[i]
        norm = _tensortype(node.outputs[0])(norm)
        mapLabel = _string_builder(node.type)
        mapParams = params
        mapDims = dims
        mapEntry, mapExit = self.state.add_map(mapLabel, dict(zip(mapParams, mapDims)))
        tasklet = self.state.add_tasklet(
            mapLabel, {"j0"}, {"out"}, "out = j0/" + str(norm)
        )
        self.add_in_memlets([inputNode], mapEntry, tasklet, [dims], [params])
        outputShape = _tensorshape(node.outputs[0])
        if node.get_attr("keep_dims"):
            outputParams = [
                params[i] if outputShape[i] != 1 else "0" for i in range(len(mapParams))
            ]
        else:
            temp = set(mapParams[a] for a in reduction_axes)
            outputParams = list(set(mapParams) - temp)
            outputParams.sort()
        if len(outputParams) == 0:
            outputParams = ["0"]
        self.add_out_memlets(
            [outputNode],
            mapExit,
            tasklet,
            [outputDims],
            [outputParams],
            wcr="lambda a,b: a+b",
            wcr_identity=0,
        )

    # Would reduce all but the last dimension in the input.
    def visit_FusedBatchNorm(self, node):
        import math
        import numpy as np

        local_ctr = str(next(_atomic_count))
        ######### All the nodes and constants ##########
        training = node.get_attr("is_training")
        inpTensorNode, inpTensorParams, inpTensorDims = self.create_and_add_input_node(
            node.inputs[0]
        )
        scale, _, scaleDims = self.create_and_add_input_node(node.inputs[1])
        offset, _, offsetDims = self.create_and_add_input_node(node.inputs[2])
        epsilon = node.get_attr("epsilon")
        epsilon = float(epsilon)
        outputList = self.create_and_add_output_node(node)
        normalisedTensorNode = outputList[0]
        normalisationScalar = 1
        for i in inpTensorNode.desc(self.graph).shape[:-1]:
            normalisationScalar *= i
        normalisationScalar = float(normalisationScalar)
        hack_variance = self.state.add_scalar(
            "variance_sqrt" + local_ctr,
            dace.typeclass(np.float32),
            transient=True,
            toplevel=False,
        )
        ######### Maps common for both training and inference #########
        outerMapLabel = _string_builder("outer_iteration_channels")
        outerMapEntry, outerMapExit = self.state.add_map(
            outerMapLabel, dict(zip([inpTensorParams[-1]], [str(inpTensorDims[-1])]))
        )
        fbnormMapLabel = _string_builder("iteration_all_dimensions_3")
        fbnormMapEntry, fbnormMapExit = self.state.add_map(
            fbnormMapLabel, dict(zip(inpTensorParams[:-1], inpTensorDims[:-1]))
        )
        ######### Common tasklets #########
        fbnormTasklet = self.state.add_tasklet(
            "fbn_eltwise_norm",
            {"inp", "gamma", "beta", "mean", "variance"},
            {"out"},
            "out=float(double(gamma)*((double(inp)-double(mean))/double(variance))+double(beta))",
        )
        varianceTaskletSqrt = self.state.add_tasklet(
            "fbn_variance_sqrt",
            {"var"},
            {"var_prime"},
            "var_prime = math.sqrt(float(var)+float(" + str(epsilon) + "))",
        )
        ########## Common edges ##########
        inpTensorMemlet = Memlet.simple(inpTensorNode, ",".join(inpTensorParams))
        inpTensorMiddleMemlet = Memlet.simple(
            inpTensorNode, ",".join(inpTensorDims[:-1] + [inpTensorParams[-1]])
        )
        self.add_in_memlets(
            [inpTensorNode],
            outerMapEntry,
            fbnormMapEntry,
            [inpTensorDims],
            [inpTensorDims[:-1] + [inpTensorParams[-1]]],  # Jugaad
        )
        self.add_in_memlets(
            [scale, offset],
            outerMapEntry,
            fbnormMapEntry,
            [scaleDims, offsetDims],
            [[inpTensorParams[-1]], [inpTensorParams[-1]]],
        )
        self.state.add_edge(
            fbnormMapEntry,
            None,
            fbnormTasklet,
            "gamma",
            Memlet.simple(scale, inpTensorParams[-1]),
        )
        self.state.add_edge(
            fbnormMapEntry,
            None,
            fbnormTasklet,
            "beta",
            Memlet.simple(offset, inpTensorParams[-1]),
        )
        self.state.add_edge(fbnormMapEntry, None, fbnormTasklet, "inp", inpTensorMemlet)
        self.add_out_memlets(
            [normalisedTensorNode],
            outerMapExit,
            fbnormMapExit,
            [inpTensorDims],
            [inpTensorDims[:-1] + [inpTensorParams[-1]]],
        )
        self.state.add_edge(
            fbnormTasklet,
            "out",
            fbnormMapExit,
            None,
            Memlet.simple(normalisedTensorNode, ",".join(inpTensorParams)),
        )
        self.state.add_edge(
            varianceTaskletSqrt,
            "var_prime",
            hack_variance,
            None,
            Memlet.simple(hack_variance, ",".join(["0"])),
        )
        self.state.add_edge(
            hack_variance,
            None,
            fbnormMapEntry,
            None,
            Memlet.simple(hack_variance, ",".join(["0"])),
        )
        self.state.add_edge(
            fbnormMapEntry,
            None,
            fbnormTasklet,
            "variance",
            Memlet.simple(hack_variance, ",".join(["0"])),
        )
        if training:
            meanTensorNode = outputList[1]
            meanDims = self.get_default_dims(node.outputs[1])
            varianceTensorNode = outputList[2]
            tempNodeMean = self.state.add_scalar(
                _string_builder("temp_mean_scalar" + local_ctr),
                dace.typeclass(_tensortype(node.outputs[1])),
                transient=True,
                toplevel=False,
            )
            tempNodeMean_sum = self.state.add_scalar(
                _string_builder("temp_mean_sum_scalar" + local_ctr),
                dace.typeclass(_tensortype(node.outputs[1])),
                transient=True,
                toplevel=False,
            )
            tempNodeVar = self.state.add_scalar(
                _string_builder("temp_variance_scalar" + local_ctr),
                dace.typeclass(_tensortype(node.outputs[1])),
                transient=True,
                toplevel=False,
            )
            tempNodeVar_sum = self.state.add_scalar(
                _string_builder("temp_variance_sum_scalar" + local_ctr),
                dace.typeclass(_tensortype(node.outputs[1])),
                transient=True,
                toplevel=False,
            )
            # self.reinitCR(tempNodeMean_sum, "", "0", "0")
            # self.reinitCR(tempNodeVar_sum, "", "0", "0")
            ######### Mean/Variance computation maps #########
            meanMapLabel = _string_builder("iteration_all_dimensions_1")
            meanMapEntry, meanMapExit = self.state.add_map(
                meanMapLabel, dict(zip(inpTensorParams[:-1], inpTensorDims[:-1]))
            )
            varianceMapLabel = _string_builder("iteration_all_dimensions_2")
            varianceMapEntry, varianceMapExit = self.state.add_map(
                varianceMapLabel, dict(zip(inpTensorParams[:-1], inpTensorDims[:-1]))
            )
            ######### Tasklets for mean and variance #########
            meanTaskletSum = self.state.add_tasklet(
                "fbn_mean_computation_sum", {"j0"}, {"out"}, "out = j0"
            )
            meanTaskletNorm = self.state.add_tasklet(
                "fbn_mean_computation_norm",
                {"out"},
                {"out_prime"},
                "out_prime = out/" + str(normalisationScalar),
            )
            varianceTaskletSum = self.state.add_tasklet(
                "fbn_variance_compation_sum",
                {"inp0", "inp1"},  # inp0 is the mean, inp1 is the input
                {"out"},
                "out = (inp1 - inp0) * (inp1 - inp0)",
            )
            varianceTaskletNorm = self.state.add_tasklet(
                "fbn_variance_computation_norm",
                {"out"},
                {"out_prime"},
                "out_prime = out/" + str(normalisationScalar),
            )
            ########## All the edges ##########
            self.state.add_edge(
                outerMapEntry, None, meanMapEntry, None, inpTensorMiddleMemlet
            )
            self.state.add_edge(
                meanMapEntry, None, meanTaskletSum, "j0", inpTensorMemlet
            )
            self.state.add_edge(
                meanTaskletSum,
                "out",
                meanMapExit,
                None,
                Memlet.simple(
                    tempNodeMean_sum, "0", wcr_str="lambda a, b: a+b", wcr_identity=0
                ),
            )
            self.state.add_edge(
                meanMapExit,
                None,
                tempNodeMean_sum,
                None,
                Memlet.simple(
                    tempNodeMean_sum, "0", wcr_str="lambda a, b: a+b", wcr_identity=0
                ),
            )
            self.state.add_edge(
                tempNodeMean_sum,
                None,
                meanTaskletNorm,
                "out",
                Memlet.simple(tempNodeMean_sum, "0"),
            )
            self.state.add_edge(
                meanTaskletNorm,
                "out_prime",
                tempNodeMean,
                None,
                Memlet.simple(tempNodeMean, "0"),
            )
            self.add_in_memlets(
                [tempNodeMean],
                varianceMapEntry,
                varianceTaskletSum,
                [["0"]],
                [["0"]],
                identifier="inp",
            )
            self.state.add_edge(
                outerMapEntry, None, varianceMapEntry, None, inpTensorMiddleMemlet
            )
            self.state.add_edge(
                varianceMapEntry, None, varianceTaskletSum, "inp1", inpTensorMemlet
            )
            self.add_out_memlets(
                [tempNodeVar_sum],
                varianceMapExit,
                varianceTaskletSum,
                [["0"]],
                [["0"]],
                wcr="lambda a,b: a+b",
                wcr_identity=0,
            )
            self.state.add_edge(
                tempNodeVar_sum,
                None,
                varianceTaskletNorm,
                "out",
                Memlet.simple(tempNodeVar_sum, "0"),
            )
            self.state.add_edge(
                varianceTaskletNorm,
                "out_prime",
                tempNodeVar,
                None,
                Memlet.simple(tempNodeVar, "0"),
            )
            hack_mean = self.state.add_read("temp_mean_scalar" + local_ctr)  # jugaad 2
            self.state.add_edge(
                tempNodeMean,
                None,
                hack_mean,
                None,
                Memlet.simple(hack_mean, ",".join(["0"])),
            )
            self.state.add_edge(
                hack_mean,
                None,
                fbnormMapEntry,
                None,
                Memlet.simple(hack_mean, ",".join(["0"])),
            )
            self.state.add_edge(
                fbnormMapEntry,
                None,
                fbnormTasklet,
                "mean",
                Memlet.simple(hack_mean, ",".join(["0"])),
            )

            self.state.add_edge(
                tempNodeVar,
                None,
                varianceTaskletSqrt,
                "var",
                Memlet.simple(tempNodeVar, ",".join(["0"])),
            )
            hack_mean_1 = self.state.add_read(
                "temp_mean_scalar" + local_ctr
            )  # jugaad 2
            self.state.add_edge(
                tempNodeMean,
                None,
                hack_mean_1,
                None,
                Memlet.simple(hack_mean_1, ",".join(["0"])),
            )
            self.state.add_edge(
                hack_mean_1,
                None,
                outerMapExit,
                None,
                Memlet.simple(meanTensorNode, ",".join([inpTensorParams[-1]])),
            )
            self.state.add_edge(
                outerMapExit,
                None,
                meanTensorNode,
                None,
                Memlet.simple(meanTensorNode, ",".join(meanDims)),
            )
            hack_variance_1 = self.state.add_read(
                "temp_variance_scalar" + local_ctr
            )  # jugaad 2
            self.state.add_edge(
                tempNodeVar,
                None,
                hack_variance_1,
                None,
                Memlet.simple(hack_variance_1, ",".join(["0"])),
            )
            self.state.add_edge(
                hack_variance_1,
                None,
                outerMapExit,
                None,
                Memlet.simple(varianceTensorNode, ",".join([inpTensorParams[-1]])),
            )
            self.state.add_edge(
                outerMapExit,
                None,
                varianceTensorNode,
                None,
                Memlet.simple(varianceTensorNode, ",".join(meanDims)),
            )
            # Caching value for backpass
            varSqrtRead = self.state.add_read("variance_sqrt" + local_ctr)
            self.state.add_edge(
                hack_variance,
                None,
                varSqrtRead,
                None,
                Memlet.simple(hack_variance, "0"),
            )
            self.add_out_memlets(
                [outputList[4]],
                outerMapExit,
                varSqrtRead,
                [meanDims],
                [[inpTensorParams[-1]]],
            )
            self.state.add_edge(
                meanTensorNode,
                None,
                outputList[3],
                None,
                Memlet.simple(meanTensorNode, ",".join(meanDims)),
            )

        else:
            # Input and output edges have been added through the two maps and
            # the tasklets. Need to add variance square root edges and edges
            # between mean and tasklet. Maybe this can be factored out...
            populationMeanTensor, populationMeanParams, populationMeanDims = self.create_and_add_input_node(
                node.inputs[3]
            )
            populationVarianceTensor, populationVarianceParams, populationVarianceDims = self.create_and_add_input_node(
                node.inputs[4]
            )
            self.add_in_memlets(
                [populationMeanTensor],
                outerMapEntry,
                fbnormMapEntry,
                [populationMeanDims],
                [[inpTensorParams[-1]]],
            )
            self.state.add_edge(
                fbnormMapEntry,
                None,
                fbnormTasklet,
                "mean",
                Memlet.simple(populationMeanTensor, ",".join([inpTensorParams[-1]])),
            )
            self.state.add_edge(
                populationVarianceTensor,
                None,
                outerMapEntry,
                None,
                Memlet.simple(
                    populationVarianceTensor, ",".join(populationVarianceDims)
                ),
            )
            self.state.add_edge(
                outerMapEntry,
                None,
                varianceTaskletSqrt,
                "var",
                Memlet.simple(
                    populationVarianceTensor, ",".join([inpTensorParams[-1]])
                ),
            )

    def visit_FusedBatchNormGrad(self, node):
        local_ctr = str(next(_atomic_count))
        ############################INPUTS##############################################
        backpropGradients, backpropParams, backpropDims = self.create_and_add_input_node(
            node.inputs[0]
        )
        inputData, inputParams, inputDims = self.create_and_add_input_node(
            node.inputs[1]
        )
        gammaNode, _, gammaDims = self.create_and_add_input_node(node.inputs[2])
        meanNode, _, meanDims = self.create_and_add_input_node(node.inputs[3])
        stdevNode, _, stdevDims = self.create_and_add_input_node(node.inputs[4])
        #############################OUTPUTS#############################################
        outputList = self.create_and_add_output_node(node)
        imageGrads = outputList[0]
        gammaGrads = outputList[1]
        betaGrads = outputList[2]
        ############################TRANSIENTS##########################################
        gammaPrime = self.state.add_scalar(
            "gamma_prime" + local_ctr,
            _tensortype(node.outputs[1]),
            transient=True,
            toplevel=False,
        )
        betaPrime = self.state.add_array(
            "beta_prime" + local_ctr,
            [1],
            _tensortype(node.outputs[2]),
            transient=True,
            toplevel=False,
        )
        read_betaprime = self.state.add_read("beta_prime" + local_ctr)
        ###############################MAPS##############################################
        channelMapLabel = _string_builder(node.type) + "_outer"
        channelMapEntry, channelMapExit = self.state.add_map(
            channelMapLabel, dict(zip([backpropParams[-1]], [backpropDims[-1]]))
        )
        innerMap1Label = _string_builder(node.type) + "_inner1"
        innerMap1Entry, innerMap1Exit = self.state.add_map(
            innerMap1Label, dict(zip(backpropParams[:-1], backpropDims[:-1]))
        )
        innerMap2Label = _string_builder(node.type) + "_inner2"
        innerMap2Entry, innerMap2Exit = self.state.add_map(
            innerMap2Label, dict(zip(backpropParams[:-1], backpropDims[:-1]))
        )
        #############################TASKLETS###########################################
        nhw = 1
        for i in backpropGradients.desc(self.graph).shape[:-1]:
            nhw *= i
        nhw = str(float(nhw))
        auxGradsTasklet = self.state.add_tasklet(
            "linear_grads",
            {"y_prime", "x", "mu", "stdev"},
            {"gamma_prime", "beta_prime"},
            "beta_prime = y_prime; gamma_prime = y_prime * (x - mu) / stdev;",
        )
        # add inconnector beta_prime
        inputGradsTasklet = self.state.add_tasklet(
            "input_grads",
            {"gamma", "gamma_prime", "beta_prime", "y_prime", "x", "mu", "stdev"},
            {"x_prime"},
            "x_prime = float(gamma*("
            + nhw
            + "*y_prime - beta_prime - (gamma_prime*(x - mu)/stdev))/(stdev*"
            + nhw
            + "));",
        )
        inputs = [backpropGradients, inputData, meanNode, stdevNode]
        dims = [backpropDims, inputDims, meanDims, stdevDims]
        middleParams = [
            backpropDims[:-1] + [backpropParams[-1]],
            backpropDims[:-1] + [backpropParams[-1]],
            [backpropParams[-1]],
            [backpropParams[-1]],
        ]
        # auxGradTasklet in-edges
        self.add_in_memlets(inputs, channelMapEntry, innerMap1Entry, dims, middleParams)
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "y_prime",
            Memlet.simple(backpropGradients, ",".join(backpropParams)),
        )
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "x",
            Memlet.simple(inputData, ",".join(inputParams)),
        )
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "mu",
            Memlet.simple(meanNode, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "stdev",
            Memlet.simple(stdevNode, ",".join([backpropParams[-1]])),
        )
        # auxGradsTasklet out-edges
        self.state.add_edge(
            auxGradsTasklet,
            "gamma_prime",
            innerMap1Exit,
            None,
            Memlet.simple(
                gammaPrime, "0", wcr_str="lambda a,b: a+b", wcr_identity=float(0)
            ),
        )
        self.state.add_edge(
            innerMap1Exit,
            None,
            gammaPrime,
            None,
            Memlet.simple(
                gammaPrime, "0", wcr_str="lambda a,b: a+b", wcr_identity=float(0)
            ),
        )
        self.state.add_edge(
            auxGradsTasklet,
            "beta_prime",
            innerMap1Exit,
            None,
            Memlet.simple(
                betaPrime, "0", wcr_str="lambda a, b: a+b", wcr_identity=float(0)
            ),
        )
        self.state.add_edge(
            innerMap1Exit,
            None,
            betaPrime,
            None,
            Memlet.simple(
                betaPrime, "0", wcr_str="lambda a, b: a+b", wcr_identity=float(0)
            ),
        )
        # second map in-edges
        self.add_in_memlets(
            [gammaNode],
            channelMapEntry,
            innerMap2Entry,
            [gammaDims],
            [[backpropParams[-1]]],
        )
        for node, param in zip(inputs, middleParams):
            self.state.add_edge(
                channelMapEntry,
                None,
                innerMap2Entry,
                None,
                Memlet.simple(node, ",".join(param)),
            )
        self.state.add_edge(
            gammaPrime, None, innerMap2Entry, None, Memlet.simple(gammaPrime, "0")
        )
        self.state.add_edge(
            read_betaprime, None, innerMap2Entry, None, Memlet.simple(betaPrime, "0")
        )
        # inputGradsTasklet in-edges
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "gamma",
            Memlet.simple(gammaNode, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "beta_prime",
            Memlet.simple(read_betaprime, "0"),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "gamma_prime",
            Memlet.simple(gammaPrime, "0"),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "y_prime",
            Memlet.simple(backpropGradients, ",".join(backpropParams)),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "mu",
            Memlet.simple(meanNode, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "x",
            Memlet.simple(inputData, ",".join(inputParams)),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "stdev",
            Memlet.simple(stdevNode, ",".join([backpropParams[-1]])),
        )
        # inputGradsTasklet out-edges
        self.state.add_edge(
            inputGradsTasklet,
            "x_prime",
            innerMap2Exit,
            None,
            Memlet.simple(imageGrads, ",".join(backpropParams)),
        )
        self.add_out_memlets(
            [imageGrads],
            channelMapExit,
            innerMap2Exit,
            [backpropDims],
            [backpropDims[:-1] + [backpropParams[-1]]],
        )
        # Add reads and edges. Can't directly add out memlets.
        read_gammaprime = self.state.add_read("gamma_prime" + local_ctr)
        self.state.add_edge(
            gammaPrime, None, read_gammaprime, None, Memlet.simple(gammaPrime, "0")
        )
        self.state.add_edge(
            betaPrime, None, read_betaprime, None, Memlet.simple(betaPrime, "0")
        )
        self.add_out_memlets(
            [gammaGrads],
            channelMapExit,
            read_gammaprime,
            [gammaDims],
            [[backpropParams[-1]]],
        )
        self.add_out_memlets(
            [betaGrads], channelMapExit, betaPrime, [gammaDims], [[backpropParams[-1]]]
        )

    def visit_Tile(self, node):
        # Replicates input multiple times
        inputList = []
        inputNodes = []

        state = self.state

        for inp in node.inputs:

            label = _string_builder(inp.name)
            try:
                inputNode = state.find_node(label)
            except (LookupError):

                inputNode = self.create_and_add_input_node(inp)[0]

            inputNodes.append(inputNode)
            inputList.append(inputNode.desc(self.graph))

        outputList = self.create_and_add_output_node(node)

        mapLabel = _string_builder(node.type)
        outputDims = self.get_default_dims(node.outputs[0])
        outputParams = self.get_default_params(node.outputs[0])
        inputDims = self.get_default_dims(node.inputs[0])
        inputParams = []

        for i, dim in enumerate(inputList[0].shape):
            inputParams.append("i" + str(i) + "%" + str(dim))

        mapDict = dict(zip(outputParams, outputDims))
        inMemletDict = dict(j0=Memlet.simple(inputNodes[0], ",".join(inputParams)))
        outMemletDict = dict(out=Memlet.simple(outputList[0], ",".join(outputParams)))
        code = "out = j0"
        tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            mapLabel, mapDict, inMemletDict, code, outMemletDict
        )
        state.add_edge(
            inputNodes[0],
            None,
            map_entry,
            None,
            Memlet.simple(inputNodes[0], ",".join(inputDims)),
        )
        state.add_edge(
            map_exit,
            None,
            outputList[0],
            None,
            Memlet.simple(outputList[0], ",".join(outputDims)),
        )

    def visit_ReadVariableOp(self, node):
        # TODO this should ideally be an add_read on the input name
        state = self.state
        inp = node.inputs[0]
        label = _string_builder(inp.name)
        try:
            inputNode = state.find_node(label)
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.outputs[0]))
            shape = dace.properties.ShapeProperty.from_string(
                str(_tensorshape(node.outputs[0]))
            )
            inputNode = state.add_transient(name=label, shape=shape, dtype=dtype)

        outputNode = self.create_and_add_output_node(node)[0]
        outputDims = self.get_default_dims(node.outputs[0])
        self.state.add_edge(
            inputNode,
            None,
            outputNode,
            None,
            Memlet.simple(outputNode, ",".join(outputDims)),
        )

    def visit_VarHandleOp(self, node):
        self.create_and_add_output_node(node)

    def visit_PreventGradient(self, node):
        # Just a memcopy, works like visit_assign or visit_identity
        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            # relevant input is at position 0
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)

        for count, out in enumerate(node.outputs):

            dims = self.get_default_dims(out)
            params = self.get_default_params(out)
            outputParams.append(params)
            outputDims.append(dims)

        memlet = Memlet.simple(inputNodes[0], ",".join(inputDims[0]))
        state.add_edge(inputNodes[0], None, outputList[0], None, memlet)

    def visit_ExpandDims(self, node):
        # Takes an N-dimensional array and adds one dimension to it with a
        # length of 1. Example: (M,K) -> (1,M,K).
        # We can just use DaCe memory copy to do the same
        state = self.state
        inputList = []
        inputNodes = []
        inputDims = []
        inputParams = []

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputDims.append(dims)
                inputParams.append(params)

        outputList = self.create_and_add_output_node(node)
        memlet = Memlet.simple(inputNodes[0], ",".join(inputDims[0]))
        state.add_edge(inputNodes[0], None, outputList[0], None, memlet)

    def visit_ApplyGradientDescent(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        mapParams = []
        mapRange = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):

            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputParams.append(params)
            inputDims.append(dims)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        mapLabel = _string_builder(node.type)
        # inputList[1] is learning rate which needs its own parameter
        inputParams[1] = ["i4"]
        # This is the variable which is input and output of this map at the same
        # time. We create the output version of it here
        out = node.inputs[0]
        outName = _string_builder(out.name)
        outputNode = self.state.add_write(outName)
        dims = self.get_default_dims(out)
        params = self.get_default_params(out)
        outputList = [outputNode]
        outputParams = [params]
        outputDims = [dims]

        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0] + ["i4"]
        mapRange = inputDims[0] + ["0:1"]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(
            mapLabel, {"j0", "j1", "j2"}, {"out"}, "out = j0-(j1*j2)"
        )
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)

    def visit_ResourceApplyGradientDescent(self, node):
        # this is actually the same as above, but the real input has no shape or type.
        # that has to be changed.
        state = self.state
        inputList = []
        inputNodes = []
        inputParams = []
        inputDims = []

        # make the input node using the gradient node, because the input node has type "resource"
        # and no shape information.
        inp = node.inputs[0]
        label = _string_builder(inp.name)
        try:
            inputNode = state.find_node(label)
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.inputs[2]))
            shape = dace.properties.ShapeProperty.from_string(
                str(_tensorshape(node.inputs[2]))
            )
            inputNode = state.add_transient(name=label, shape=shape, dtype=dtype)
        inputNodes.append(inputNode)
        inputParams.append(self.get_default_params(node.inputs[2]))
        inputDims.append(self.get_default_dims(node.inputs[2]))

        for count, inp in enumerate(node.inputs):
            if count == 0:
                continue
            else:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputParams.append(params)
                inputDims.append(dims)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)

        # inputList[1] is learning rate which needs its own parameter
        inputParams[1] = ["i4"]
        out = node.inputs[2]
        outName = _string_builder(node.inputs[0].name)
        outputNode = state.add_write(outName)
        dims = self.get_default_dims(out)
        params = self.get_default_params(out)
        outputList = [outputNode]
        outputParams = [params]
        outputDims = [dims]
        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0] + ["i4"]
        mapRange = inputDims[0] + ["0:1"]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(
            mapLabel, {"j0", "j1", "j2"}, {"out"}, "out = j0-(j1*j2)"
        )
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)

    def visit_MatMul(self, node):
        # 2d Matrix Multiplication
        inputList = []
        inputNodes = []
        state = self.state
        mapParams = []
        outputParams = [[]]
        mapRange = []
        outputDims = [[]]
        inputParams = [[], []]
        inputDims = [[], []]

        for inp in node.inputs:
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)

        ndims = len(outputList[0].desc(self.graph).shape)
        # Params for higher dimensions (not verified)
        # (for 2d it works)
        for i in range(0, ndims + 1):
            if i == ndims:
                mapParams.append("i" + str(i))
                inputParams[1].append("i" + str(i))
                outputParams[0].append("i" + str(i))

            elif i == ndims - 1:
                mapParams.append("i" + str(i))
                inputParams[0].append("i" + str(i))
                inputParams[1].append("i" + str(i))

            elif i == ndims - 2:
                mapParams.append("i" + str(i))
                inputParams[0].append("i" + str(i))
                outputParams[0].append("i" + str(i))

            else:
                mapParams.append("i" + str(i))
                inputParams[0].append("i" + str(i))
                inputParams[1].append("i" + str(i))
                outputParams[0].append("i" + str(i))

        for i in range(0, ndims):
            inputDims[0].append(str(0) + ":" + str(node.inputs[0].shape[i]))
            inputDims[1].append(str(0) + ":" + str(node.inputs[1].shape[i]))
            outputDims[0].append(str(0) + ":" + str(node.outputs[0].shape[i]))
            mapRange.append(str(0) + ":" + str(node.inputs[0].shape[i]))

        mapRange.append(str(0) + ":" + str(node.outputs[0].shape[ndims - 1]))
        # if first input needs to be transposed
        if node.get_attr("transpose_a"):
            mapRange[0], mapRange[1] = mapRange[1], mapRange[0]
            inputParams[0][0], inputParams[0][1] = inputParams[0][1], inputParams[0][0]
        # if second input needs to be transposed
        if node.get_attr("transpose_b"):
            inputParams[1][0], inputParams[1][1] = inputParams[1][1], inputParams[1][0]

        mentry, mexit = state.add_map(
            "matmul_outer", {mapParams[1]: mapRange[1]}, dace.ScheduleType.Sequential
        )
        minentry, minexit = state.add_map(
            "matmul_inner",
            {mapParams[0]: mapRange[0], mapParams[2]: mapRange[2]},
            dace.ScheduleType.CPU_Multicore,
        )
        tasklet = state.add_tasklet("mm_code", {"j0", "j1"}, {"out"}, "out = j0*j1")

        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(minentry, None, tasklet, name, memlet)

        for i, out in enumerate(outputList):
            name = "out"
            memlet = Memlet.simple(
                out,
                ",".join(outputParams[i]),
                wcr_str="lambda a,b: a+b",
                wcr_identity=0,
            )
            state.add_edge(tasklet, name, minexit, None, memlet)

        self.reinitCR(outputList[0], outputParams, outputDims, "0")
        self.add_out_memlets(
            outputList, mexit, minexit, outputDims, outputParams, "lambda a,b: a+b", 0
        )
        self.add_in_memlets(inputNodes, mentry, minentry, inputDims, inputParams)

    def visit_element_wise_op(self, node, operation):
        """ Handles all the element wise operations, supports broadcasting. """
        inputList = []
        inputNodes = []
        mapParams = []
        outputParams = []
        mapRange = []
        outputDims = []
        inputParams = []
        inputDims = []
        state = self.state

        for inp in node.inputs:

            inputNode, _, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)

        outputNodes = self.create_and_add_output_node(node)
        mapLabel = _string_builder(node.type)
        # create params
        for inp in inputList:
            inputParamsString = []
            for i, dim in enumerate(inp.shape):
                # scalar case that we want to broadcast
                if str(dim) == "1":
                    inputParamsString.append("0")
                else:
                    inputParamsString.append("i" + str(i))

            inputParams.append(inputParamsString)

        params = self.get_default_params(node.outputs[0])
        dims = self.get_default_dims(node.outputs[0])
        outputParams.append(params)
        outputDims.append(dims)

        mapParams = outputParams[0]
        mapRange = outputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(
            mapLabel, {"j0", "j1"}, {"out"}, "out = j0 " + operation + " j1"
        )
        self.add_out_memlets(outputNodes, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Conv2D(self, node):
        inputList = []
        inputNodes = []
        ndims = 0
        strides = node.get_attr("strides")[1]
        state = self.state

        for inp in node.inputs:
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)
        ndims = len(outputList[0].desc(self.graph).shape)
        mapLabel = _string_builder(node.type)

        mapParams = []
        outputParams = []
        mapRange = []
        outputDims = [[]]
        inputParams = []
        inputDims = [[], []]
        # create conv params
        inputParams.append(
            ["i0", "i1*" + str(strides) + "+i5", "i2*" + str(strides) + "+i6", "i3"]
        )
        inputParams.append(["i5", "i6", "i3", "i4"])
        outputParams.append(["i0", "i1", "i2", "i4"])
        # create conv dims
        for i in range(0, ndims):
            inputDims[0].append(str(0) + ":" + str(node.inputs[0].shape[i]))
            inputDims[1].append(str(0) + ":" + str(node.inputs[1].shape[i]))
            outputDims[0].append(str(0) + ":" + str(node.outputs[0].shape[i]))
        # add a padding map for same padding(zero padding so that input and
        # output of convolution have the same size)
        if str(node.get_attr("padding"))[2:-1] == "SAME":
            paddedInput, paddedDims = self.inputPadding(
                node,
                inputNodes[0],
                inputList[0],
                outputList[0].desc(self.graph).shape[1],
                inputList[1].shape[0],
                strides,
                inputDims[0],
            )
            inputDims[0] = paddedDims
            inputNodes[0] = paddedInput

        mapParams = outputParams[0]
        mapParams2 = inputParams[1][:-1]
        mapRange = outputDims[0]
        mapRange2 = inputDims[1][:-1]

        mapEntry, mapExit = state.add_map(
            mapLabel + "_outer", dict(zip(mapParams, mapRange))
        )
        mapEntry2, mapExit2 = state.add_map(
            mapLabel + "_inner", dict(zip(mapParams2, mapRange2))
        )
        self.reinitCR(outputList[0], outputParams, outputDims, "0")
        tasklet = state.add_tasklet(
            mapLabel, {"j0", "j1"}, {"out"}, "out = j0 * j1;"
        )  # printf(\"%f\\t\", j0);")
        self.add_out_memlets(
            outputList,
            mapExit,
            mapExit2,
            outputDims,
            outputParams,
            "lambda a,b: a+b",
            0,
        )
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
        # add memlets from inner map to tasklet
        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)
        # add memelets from tasklet to cr
        for i, out in enumerate(outputList):
            name = "out"
            memlet = Memlet.simple(
                out,
                ",".join(outputParams[i]),
                wcr_str="lambda a,b: a+b",
                wcr_identity=0,
            )
            state.add_edge(tasklet, name, mapExit2, None, memlet)

    def visit_BiasAdd(self, node):

        inputList = []
        inputNodes = []
        state = self.state

        for inp in node.inputs:
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)
        dims = outputList[0].desc(self.graph).shape

        mapLabel = _string_builder(node.type)
        mapParams = []
        outputParams = []
        mapRange = []
        outputDims = []
        inputParams = [[], []]
        inputDims = [[], []]

        params = self.get_default_params(node.outputs[0])
        dims = self.get_default_dims(node.outputs[0])
        outputParams.append(params)
        outputDims.append(dims)

        mapParams = outputParams[0]
        inputParams[0] = outputParams[0]
        # the bias matches the last dimension of input resp. output
        inputParams[1] = [mapParams[-1]]
        mapRange = outputDims[0]
        inputDims[0] = outputDims[0]
        inputDims[1] = ["0:" + str(node.inputs[1].shape[0])]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0 + j1")
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_MaxPool(self, node):
        inputList = []
        inputNodes = []
        dims = []
        inputDims = []
        strides_0 = node.get_attr("strides")[1]
        strides_1 = node.get_attr("strides")[2]
        ksize_0 = node.get_attr("ksize")[1]
        ksize_1 = node.get_attr("ksize")[2]
        state = self.state

        for inp in node.inputs:
            inputNode, _, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)
        inputParams = [
            ["i0", "i1*" + str(strides_0) + "+i4", "i2*" + str(strides_1) + "+i5", "i3"]
        ]

        outputParams = []
        outputDims = []
        outputList = self.create_and_add_output_node(node)
        dims = self.get_default_dims(node.outputs[0])
        params = self.get_default_params(node.outputs[0])
        outputDims.append(dims)
        outputParams.append(params)

        if str(node.get_attr("padding"))[2:-1] == "SAME":
            assert ksize_0 == ksize_1
            assert strides_0 == strides_1
            paddedInput, paddedDims = self.inputPadding(
                node,
                inputNodes[0],
                inputList[0],
                outputList[0].desc(self.graph).shape[1],
                ksize_0,
                strides_0,
                inputDims[0],
            )
            inputDims[0] = paddedDims
            inputNodes[0] = paddedInput

        mapLabel = _string_builder(node.type)
        mapParams1 = outputParams[0]
        mapRange1 = outputDims[0]
        mapParams2 = ["i4", "i5"]
        mapRange2 = ["0:" + str(ksize_0), "0:" + str(ksize_1)]

        mapEntry, mapExit = state.add_map(
            mapLabel + "_outer", dict(zip(mapParams1, mapRange1))
        )
        mapEntry2, mapExit2 = state.add_map(
            mapLabel + "_inner", dict(zip(mapParams2, mapRange2))
        )
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.reinitCR(outputList[0], outputParams, outputDims, "-99999999999")
        self.add_out_memlets(
            outputList,
            mapExit,
            mapExit2,
            outputDims,
            outputParams,
            "lambda a, b: max(a,b)",
            -99999999999,
        )
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
        # add memlets from inner map to tasklet
        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)
        # add memelets from tasklet to cr
        for i, out in enumerate(outputList):
            name = "out"
            memlet = Memlet.simple(
                out,
                ",".join(outputParams[i]),
                wcr_str="lambda a, b: max(a,b)",
                wcr_identity=-99999999999,
            )
            state.add_edge(tasklet, name, mapExit2, None, memlet)

    # TODO bugfix with padding, fails for cases where padding is on left and
    # right, and up and down. Will have to rewrite expression for
    # normalisationScalar
    def visit_AvgPool(self, node):
        inputList = []
        inputNodes = []
        inputDims = []
        strides_0 = node.get_attr("strides")[1]
        strides_1 = node.get_attr("strides")[2]
        ksize_0 = node.get_attr("ksize")[1]
        ksize_1 = node.get_attr("ksize")[2]
        state = self.state
        local_count = str(next(_atomic_count))
        for inp in node.inputs:
            inputNode, _, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)
        inputParams = [
            ["i0", "i1*" + str(strides_0) + "+i4", "i2*" + str(strides_1) + "+i5", "i3"]
        ]

        outputParams = []
        outputDims = []
        outputList = self.create_and_add_output_node(node)
        dims = self.get_default_dims(node.outputs[0])
        params = self.get_default_params(node.outputs[0])
        outputDims.append(dims)
        outputParams.append(params)

        assert str(node.get_attr("padding"))[2:-1] == "VALID"
        # if str(node.get_attr("padding"))[2:-1] == "SAME":
        #    assert ksize_0 == ksize_1
        #    assert strides_0 == strides_1
        #    paddedInput, paddedDims = self.inputPadding(
        #        node,
        #        inputNodes[0],
        #        inputList[0],
        #        outputList[0].desc(self.graph).shape[1],
        #        ksize_0,
        #        strides_0,
        #        inputDims[0],
        #    )
        #    inputDims[0] = paddedDims
        #    inputNodes[0] = paddedInput

        mapLabel = _string_builder(node.type)
        mapParams1 = outputParams[0]
        mapRange1 = outputDims[0]
        mapParams2 = ["i4", "i5"]
        mapRange2 = ["0:" + str(ksize_0), "0:" + str(ksize_1)]

        mapEntry, mapExit = state.add_map(
            mapLabel + "_outer", dict(zip(mapParams1, mapRange1))
        )
        mapEntry2, mapExit2 = state.add_map(
            mapLabel + "_inner", dict(zip(mapParams2, mapRange2))
        )
        tasklet = state.add_tasklet(mapLabel + "_sum", {"j0"}, {"out"}, "out = j0")
        imgH = node.inputs[0].shape[1]
        imgW = node.inputs[0].shape[2]
        # normalisationScalar = "max((min({imgH}-1,{affine_Hexp}+{kernH}-1)-{affine_Hexp}+1)*(min({imgW}-1,{affine_Wexp}+{kernW}-1)-{affine_Wexp}+1),1)".format(
        #    imgH=str(imgH),
        #    imgW=str(imgW),
        #    affine_Hexp=str(strides_0) + "*" + str(mapParams1[1]),
        #    affine_Wexp=str(strides_1) + "*" + str(mapParams1[2]),
        #    kernH=str(ksize_0),
        #    kernW=str(ksize_1),
        # )
        normalisationScalar = str(ksize_0 * ksize_1)
        tasklet_norm = state.add_tasklet(
            mapLabel + "_norm",
            {"out"},
            {"out_n"},
            "out_n = out/" + normalisationScalar
            # + ';printf("%d",'
            # + normalisationScalar
            # + ");",
        )
        temp_node = self.state.add_scalar(
            "scratch_node" + local_count,
            dace.typeclass(_tensortype(node.outputs[0])),
            transient=True,
            toplevel=False,
        )
        memletTempNode = Memlet.simple(
            str(temp_node), "0", wcr_str="lambda a, b: a+b", wcr_identity=0
        )
        memletTempNode_nocr = Memlet.simple(str(temp_node), "0")
        memletOutputInner = Memlet.simple(outputList[0], ",".join(outputParams[0]))
        memletOutputOuter = Memlet.simple(outputList[0], ",".join(outputDims[0]))
        state.add_edge(mapExit2, None, temp_node, None, memletTempNode)
        state.add_edge(temp_node, None, tasklet_norm, "out", memletTempNode_nocr)
        state.add_edge(tasklet_norm, "out_n", mapExit, None, memletOutputInner)
        state.add_edge(mapExit, None, outputList[0], None, memletOutputOuter)
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
        # add memlets from inner map to tasklet
        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)
        # add memelets from tasklet to cr
        state.add_edge(tasklet, "out", mapExit2, None, memletTempNode)

    def visit_AvgPoolGrad(self, node):
        assert str(node.get_attr("padding"))[2:-1] == "VALID"
        strides_0 = node.get_attr("strides")[1]
        strides_1 = node.get_attr("strides")[2]
        ksize_0 = node.get_attr("ksize")[1]
        ksize_1 = node.get_attr("ksize")[2]
        backpropGrads, backpropParams, backpropDims = self.create_and_add_input_node(
            node.inputs[1]
        )
        outputNode = self.create_and_add_output_node(node)[0]
        outputParams = [
            "i0",
            "i1*" + str(strides_0) + "+i4",
            "i2*" + str(strides_1) + "+i5",
            "i3",
        ]
        outputDims = self.get_default_dims(node.outputs[0])
        outerMapLabel = _string_builder(node.type) + "_outer"
        outerMapParams = backpropParams
        outerMapDims = backpropDims
        outerMapEntry, outerMapExit = self.state.add_map(
            outerMapLabel, dict(zip(outerMapParams, outerMapDims))
        )
        innerMapLabel = _string_builder(node.type) + "_inner"
        innerMapParams = ["i4", "i5"]
        innerMapDims = ["0:" + str(ksize_0), "0:" + str(ksize_1)]
        innerMapEntry, innerMapExit = self.state.add_map(
            innerMapLabel, dict(zip(innerMapParams, innerMapDims))
        )
        normalisationScalar = ksize_0 * ksize_1
        tasklet = self.state.add_tasklet(
            _string_builder(node.type),
            {"backpropGrad"},
            {"outpGrad"},
            "outpGrad = backpropGrad / " + str(normalisationScalar),
        )
        self.add_in_memlets(
            [backpropGrads],
            outerMapEntry,
            innerMapEntry,
            [backpropDims],
            [backpropParams],
        )
        self.state.add_edge(
            innerMapEntry,
            None,
            tasklet,
            "backpropGrad",
            Memlet.simple(backpropGrads, ",".join(backpropParams)),
        )
        self.state.add_edge(
            tasklet,
            "outpGrad",
            innerMapExit,
            None,
            Memlet.simple(
                outputNode,
                ",".join(outputParams),
                wcr_identity=0,
                wcr_str="lambda a,b: a+b",
            ),
        )
        self.add_out_memlets(
            [outputNode],
            outerMapExit,
            innerMapExit,
            [outputDims],
            [outputParams],
            "lambda a, b: a+b",
            0,
        )

    def visit_Relu(self, node):

        inputList = []
        inputNodes = []
        state = self.state
        inputParams = []
        inputDims = []

        for inp in node.inputs:

            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)

        mapLabel = _string_builder(node.type)
        mapParams = []
        mapRange = []
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(
            mapLabel, {"j0"}, {"out"}, "out = max(dace.float32(0),j0)"
        )
        self.add_out_memlets(outputList, mapExit, tasklet, inputDims, inputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_ShapeN(self, node):
        inputList = []
        inputNodes = []
        state = self.state
        inputParams = []
        inputDims = []

        for inp in node.inputs:
            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)

        mapLabel = _string_builder(node.type)
        for i, node in enumerate(outputList):
            tasklet = state.add_tasklet(
                mapLabel + str(i),
                {},
                {"out"},
                "\n".join(
                    [
                        "out[%d] = %s" % (j, dim)
                        for j, dim in enumerate(inputList[i].shape)
                    ]
                ),
            )
            self.state.add_edge(
                tasklet,
                "out",
                node,
                None,
                Memlet.simple(node, "0:" + str(len(inputDims[i]))),
            )

    def visit_Reshape(self, node):

        inputNode, params, dims = self.create_and_add_input_node(node.inputs[0])
        outputList = self.create_and_add_output_node(node)
        outputParams = [self.get_default_params(node.outputs[0])]
        outputDims = [self.get_default_dims(node.outputs[0])]
        memlet_reshape = Memlet.simple(
            inputNode, ",".join(dims), other_subset_str=",".join(outputDims[0])
        )
        self.state.add_edge(inputNode, None, outputList[0], None, memlet_reshape)
        # state = self.state
        # inputList = []
        # inputNodes = []

        # inp = node.inputs[0]
        # inputParams = []
        # inputDims = []
        # inputNode, params, dims = self.create_and_add_input_node(inp)
        # inputParams.append(params)
        # inputDims.append(dims)
        # inDims = max(inp.shape.ndims, 1)
        # inputList.append(inputNode.desc(self.graph))
        # inputNodes.append(inputNode)

        # outputDims = []
        # outputList = self.create_and_add_output_node(node)
        # dims = outputList[0].desc(self.graph).shape
        # outDims = len(dims)
        # outputDims.append(self.get_default_dims(node.outputs[0]))

        # mapLabel = _string_builder(node.type)
        # mapParams = []
        # outputParams = [[]]
        # mapRange = []
        # mapParams = inputParams[0]
        # mapRange = inputDims[0]

        ## Reshape from 4 to 2 dimensions
        # if inDims > outDims:
        #    outputParams[0] = [
        #        "i0",
        #        "i1*"
        #        + str(node.inputs[0].shape[2])
        #        + "*"
        #        + str(node.inputs[0].shape[3])
        #        + "+i2*"
        #        + str(node.inputs[0].shape[3])
        #        + "+i3",
        #    ]
        ## Reshape from 2 to 4 dimensions
        # elif inDims < outDims:
        #    outputParams[0] = [
        #        "i0",
        #        "i1/("
        #        + str(node.outputs[0].shape[2])
        #        + "*"
        #        + str(node.outputs[0].shape[3])
        #        + ")",
        #        "(i1%"
        #        + "("
        #        + str(node.outputs[0].shape[2])
        #        + "*"
        #        + str(node.outputs[0].shape[3])
        #        + "))/"
        #        + str(node.outputs[0].shape[3]),
        #        "i1%" + str(node.outputs[0].shape[3]),
        #    ]
        ## If they have the same dimension
        # else:
        #    outputParams[0] = mapParams
        #    mapRange = outputDims[0]
        #    inputDims[0] = outputDims[0]

        # mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        # tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        # self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        # self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_MaxPoolGrad(self, node):
        # TODO: Currently only supports 2x2 maxpooling
        state = self.state
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []
        inputList = []
        inputNodes = []

        for count, inp in enumerate(node.inputs):

            inputNode, _, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            params = []

            for ndims, dim in enumerate(inp.shape):
                if (not count == 0) and (ndims == 1 or ndims == 2):
                    params.append("i" + str(ndims) + "/2")

                else:
                    params.append("i" + str(ndims))

            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        mapLabel = _string_builder(node.type)

        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(str(inputList[0].shape))

        tempNode = state.add_transient(
            _string_builder(node.name + "_tmp"), shape, dtype, toplevel=True
        )
        tempList = [tempNode]

        outputDims = inputDims
        outputParams = inputParams
        # Copy as we manipulate inputParams but don't want map params/range to
        # change
        mapParams = inputParams[0].copy()
        mapRange = inputDims[0].copy()

        mapEntry, mapExit = state.add_map(
            mapLabel + "_map1", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(
            mapLabel + "_map1",
            {"j0", "j1", "j2"},
            {"out"},
            "if (j0==j1):\n\tout = j2\nelse:\n\tout = 0",
        )

        self.add_out_memlets(tempList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

        # Second map:
        # as we don't have the indicies of the maxpooling we need to manually
        # figure out which one contributed. If it is ambigious we break the
        # tie by the following priority k[i,j]<k[i+1,j]...<k[0,j+1]...
        newDims = [inputDims[0]] * 4
        mapRange[1] += ":2"
        mapRange[2] += ":2"

        newParams = [inputParams[0]]
        # 2x2 kernel
        newParams = [
            ["i0", "i1", "i2", "i3"],
            ["i0", "i1+1", "i2", "i3"],
            ["i0", "i1", "i2+1", "i3"],
            ["i0", "i1+1", "i2+1", "i3"],
        ]

        string = """
if(j0!=0):
        out0=j0
        out1=0
        out2=0
        out3=0
elif(j1!=0):
        out0=j0
        out1=j1
        out2=0
        out3=0
elif(j2!=0):
        out0=j0
        out1=j1
        out2=j2
        out3=0
else:
        out0=j0
        out1=j1
        out2=j2
        out3=j3
"""
        tasklet = state.add_tasklet(
            mapLabel + "_map2",
            {"j0", "j1", "j2", "j3"},
            {"out0", "out1", "out2", "out3"},
            string,
        )
        mapEntry, mapExit = state.add_map(
            mapLabel + "_map2", dict(zip(mapParams, mapRange))
        )
        self.add_out_memlets(outputList * 4, mapExit, tasklet, newDims, newParams)
        self.add_in_memlets(tempList * 4, mapEntry, tasklet, newDims, newParams)

    def visit_ReluGrad(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for inp in node.inputs:
            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            dims = self.get_default_dims(out)
            params = self.get_default_params(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(
            mapLabel, {"j0", "j1"}, {"out"}, "if (j1>0):\n\tout = j0\nelse:\n\tout = 0"
        )
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_BiasAddGrad(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            outputParams.append([inputParams[0][-1]])
            outputDims.append([inputDims[0][-1]])

        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.reinitCR(outputList[0], outputParams, outputDims, "0")
        self.add_out_memlets(
            outputList, mapExit, tasklet, outputDims, outputParams, "lambda a,b: a+b", 0
        )
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Conv2DBackpropInput(self, node):
        inputList = []
        inputNodes = []
        mapParams = []
        outputParams = []
        mapRange = []
        outputDims = [[]]
        inputParams = []
        inputDims = [[], []]
        strides = node.get_attr("strides")[1]
        state = self.state

        for count, inp in enumerate(node.inputs):
            if not count == 0:
                inputNode = self.create_and_add_input_node(inp)[0]
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)

        ndims = len(outputList[0].desc(self.graph).shape)
        for i in range(0, ndims):
            inputDims[1].append(str(0) + ":" + str(inputList[1].shape[i]))
            inputDims[0].append(str(0) + ":" + str(inputList[0].shape[i]))
            outputDims[0].append(
                str(0) + ":" + str(outputList[0].desc(self.graph).shape[i])
            )

        ksize = inputList[0].shape[0]
        if str(node.get_attr("padding"))[2:-1] == "SAME":
            paddedInput, paddedDims = self.inputPadding(
                node,
                inputNodes[1],
                inputList[1],
                outputList[0].desc(self.graph).shape[1],
                ksize,
                strides,
                inputDims[1],
            )
            inputDims[1] = paddedDims
            inputNodes[1] = paddedInput
        inputParams.append(["-1-i5+" + str(ksize), "-1-i6+" + str(ksize), "i3", "i4"])
        inputParams.append(
            ["i0", "i1*" + str(strides) + "+i5", "i2*" + str(strides) + "+i6", "i4"]
        )

        outputParams.append(["i0", "i1", "i2", "i3"])

        mapLabel = _string_builder(node.type)
        mapParams = ["i0", "i1", "i2", "i3"]
        mapParams2 = ["i5", "i6", "i4"]
        mapRange = outputDims[0]
        mapRange2 = inputDims[0][:-2]
        mapRange2.append(inputDims[1][-1])
        mapEntry, mapExit = state.add_map(
            mapLabel + "_outer", dict(zip(mapParams, mapRange))
        )
        mapEntry2, mapExit2 = state.add_map(
            mapLabel + "_inner", dict(zip(mapParams2, mapRange2))
        )

        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0 * j1")
        self.reinitCR(outputList[0], outputParams, outputDims, "0")

        self.add_out_memlets(
            outputList,
            mapExit,
            mapExit2,
            outputDims,
            outputParams,
            "lambda a,b: a+b",
            0,
        )
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)
        for i, out in enumerate(outputList):
            name = "out"
            memlet = Memlet.simple(
                out,
                ",".join(outputParams[i]),
                wcr_str="lambda a,b: a+b",
                wcr_identity=0,
            )
            state.add_edge(tasklet, name, mapExit2, None, memlet)

    def visit_Conv2DBackpropFilter(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            if count != 1:
                inputNode, _, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputDims.append(dims)
        inputParams.append(["i0", "i1+i5", "i2+i6", "i3"])
        inputParams.append(["i0", "i1", "i2", "i4"])

        outputList = self.create_and_add_output_node(node)
        for count, out in enumerate(node.outputs):
            params = ["i5", "i6", "i3", "i4"]
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapParams = outputParams[0]
        mapParams2 = inputParams[1][:-1]
        mapRange = outputDims[0]
        mapRange2 = inputDims[1][:-1]
        mapLabel = _string_builder(node.type)
        mapEntry, mapExit = state.add_map(
            mapLabel + "_outer", dict(zip(mapParams, mapRange))
        )
        mapEntry2, mapExit2 = state.add_map(
            mapLabel + "_inner", dict(zip(mapParams2, mapRange2))
        )

        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0*j1")

        self.reinitCR(outputList[0], outputParams, outputDims, "0")

        self.add_out_memlets(
            outputList,
            mapExit,
            mapExit2,
            outputDims,
            outputParams,
            "lambda a,b: a+b",
            0,
        )
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)

        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)

        for i, out in enumerate(outputList):
            name = "out"
            memlet = Memlet.simple(
                out,
                ",".join(outputParams[i]),
                wcr_str="lambda a,b: a+b",
                wcr_identity=0,
            )
            state.add_edge(tasklet, name, mapExit2, None, memlet)

    def visit_SparseSoftmaxCrossEntropyWithLogits(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        inputParams = []
        inputDims = []

        for inp in node.inputs:
            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)
            inputParams.append(params)

        for out in node.outputs:
            label = _string_builder(out.name)
            try:
                outputNode = state.find_node(label)
            except (LookupError):
                dtype = dace.typeclass(_tensortype(node))
                shape = dace.properties.ShapeProperty.from_string(
                    str(_tensorshape(out))
                )
                outputNode = state.add_transient(label, shape, dtype, toplevel=True)
            outputList.append(outputNode)

        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        # 1st map, get maximum in each batchsize dimension
        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(str(inputList[1].shape))

        temp1Node = state.add_transient(
            mapLabel + "_max_tmp", shape, dtype, toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            mapLabel + "_max", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(mapLabel + "_max", {"j0"}, {"out"}, "out = j0")
        self.reinitCR(temp1Node, [inputParams[1]], [inputDims[1]], "-999999999999")
        self.add_in_memlets(
            [inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputParams[0]]
        )
        self.add_out_memlets(
            [temp1Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: max(a,b)",
            -9999999999,
        )

        # 2nd map, calculate the denominator sum
        temp2Node = state.add_transient(
            mapLabel + "_denominator_tmp", shape, dtype, toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            mapLabel + "_denominator", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(
            mapLabel + "_denominator",
            {"j0", "j1"},
            {"out"},
            "out = dace::math::exp(j0-j1);",
            language=dace.types.Language.CPP,
        )
        self.reinitCR(temp2Node, [inputParams[1]], [inputDims[1]], "0")
        inList = [inputNodes[0], temp1Node]
        self.add_in_memlets(inList, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(
            [temp2Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: a+b",
            0,
        )

        # 3rd map, calculate the sofmax
        shape = dace.properties.ShapeProperty.from_string(str(inputList[0].shape))
        temp3Node = state.add_transient(
            mapLabel + "_softmax_tmp", shape, dtype, toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            mapLabel + "_softmax", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(
            mapLabel + "_softmax",
            {"j0", "j1", "j2"},
            {"out"},
            "out = (dace::math::exp(j0-j1))/j2;",
            language=dace.types.Language.CPP,
        )
        inList = [inputNodes[0], temp1Node, temp2Node]
        paramsList = inputParams + [inputParams[1]]
        dimsList = inputDims + [inputDims[1]]
        self.add_in_memlets(inList, mapEntry, tasklet, dimsList, paramsList)
        self.add_out_memlets(
            [temp3Node], mapExit, tasklet, [inputDims[0]], [inputParams[0]]
        )

        # 4th map, calculate the cross-entropy loss for an optional loss output
        mapEntry, mapExit = state.add_map(
            mapLabel + "_loss", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(
            mapLabel + "_loss",
            {"j0", "j1"},
            {"out"},
            "if (int(j1) == i1) {\n\tout=-(dace::math::log(j0));}\nelse{\n\tout=0;}",
            language=dace.types.Language.CPP,
        )
        self.reinitCR(outputList[0], [inputParams[1]], [inputDims[1]], "0")
        self.add_in_memlets(
            [temp3Node, inputNodes[1]], mapEntry, tasklet, inputDims, inputParams
        )
        self.add_out_memlets(
            [outputList[0]],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: a+b",
            0,
        )

        # 5th map, gradient of the whole layer
        mapEntry, mapExit = state.add_map(
            mapLabel + "_gradient", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(
            mapLabel + "_gradient",
            {"j0", "j1"},
            {"out"},
            "if(int(j1)==i1):\n\tout = j0-1\nelse:\n\tout = j0",
        )
        self.add_out_memlets(
            [outputList[1]], mapExit, tasklet, [inputDims[0]], [inputParams[0]]
        )
        self.add_in_memlets(
            [temp3Node, inputNodes[1]], mapEntry, tasklet, inputDims, inputParams
        )

    def visit_Identity(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        inputParams = []
        inputDims = []

        # Create input node and its params
        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        memlet = Memlet.simple(inputNodes[0], ",".join(inputDims[0]))
        state.add_edge(inputNodes[0], None, outputList[0], None, memlet)

    def visit_LRNGrad(self, node):

        inputList = []
        inputNodes = []
        outputList = []
        state = self.state

        alpha = str(node.get_attr("alpha"))
        beta = str(node.get_attr("beta"))
        bias = str(node.get_attr("bias"))
        depth_radius = str(node.get_attr("depth_radius"))

        for count, inp in enumerate(node.inputs):
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            if count == 0:
                shortDims = []
                shortAccesses = []
                for dim in inp.shape:
                    shortDims.append("0:" + str(dim))
                    shortAccesses.append(str(dim))
                longDims = []
                longDims = shortDims + ["0:" + depth_radius + "*2+1"]
                paddedDims = []
                paddedDims += shortDims
                paddedDims[-1] += "+" + depth_radius + "*2"

        label = _string_builder(node.name)
        outputList = self.create_and_add_output_node(node)
        longParams = ["i0", "i1", "i2", "i3", "i4"]
        shortParams = ["i0", "i1", "i2", "i3"]
        copyParams = ["i0", "i1", "i2", "i3+" + depth_radius]
        normParams = ["i0", "i1", "i2", "i3+i4"]

        paddedShape = []
        paddedShape += shortAccesses
        paddedShape[-1] += "+" + depth_radius
        paddedInput = state.add_transient(
            label + "_paddedInput",
            paddedShape,
            dace.typeclass(_tensortype(node)),
            toplevel=True,
        )
        mapEntry, mapExit = state.add_map(
            label + "_padding", dict(zip(shortParams, shortDims))
        )
        tasklet = state.add_tasklet(label + "_padding", {"j0"}, {"out"}, "out=j0")
        self.add_in_memlets(
            [inputNodes[2]], mapEntry, tasklet, [shortDims], [shortParams]
        )
        self.add_out_memlets(
            [paddedInput], mapExit, tasklet, [paddedDims], [copyParams]
        )

        sqrsum = state.add_transient(
            label + "_Sqrsum", shortAccesses, _tensortype(node), toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            label + "_sqrsum", dict(zip(longParams, longDims))
        )
        tasklet = state.add_tasklet(label + "_sqrsum", {"j0"}, {"out"}, "out=j0*j0")
        self.reinitCR(sqrsum, [shortParams], [shortDims], "0")
        self.add_in_memlets(
            [paddedInput], mapEntry, tasklet, [paddedDims], [normParams]
        )
        self.add_out_memlets(
            [sqrsum], mapExit, tasklet, [shortDims], [shortParams], "lambda a,b: a+b", 0
        )

        label = _string_builder(node.name)
        norm = state.add_transient(
            label + "_Norm", shortAccesses, _tensortype(node), toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            label + "_norm", dict(zip(shortParams, shortDims))
        )
        tasklet = state.add_tasklet(
            label + "_norm", {"j0"}, {"out"}, "out=" + alpha + "*j0+" + bias
        )
        self.add_in_memlets([sqrsum], mapEntry, tasklet, [shortDims], [shortParams])
        self.add_out_memlets([norm], mapExit, tasklet, [shortDims], [shortParams])

        preOut = state.add_transient(
            label + "_preOut", shortAccesses, _tensortype(node), toplevel=True
        )
        mapEntry, mapExit = state.add_map(label, dict(zip(longParams, longDims)))
        taskletCode = (
            "if (i4=="
            + depth_radius
            + "){\n out = pow(j2,"
            + beta
            + ")-2*"
            + alpha
            + "*"
            + beta
            + "*j1*j0/j2;}\n else{\n out = -2*"
            + alpha
            + "*"
            + beta
            + "*j1*j0/j2;}"
        )
        tasklet = state.add_tasklet(
            label,
            {"j0", "j1", "j2"},
            {"out"},
            taskletCode,
            language=dace.types.Language.CPP,
        )
        self.reinitCR(preOut, [shortParams], [shortDims], "0")
        inList = [inputNodes[1]]
        inList.append(paddedInput)
        inList.append(norm)
        self.add_in_memlets(
            inList,
            mapEntry,
            tasklet,
            [shortDims, paddedDims, shortDims],
            [shortParams, normParams, shortParams],
        )
        self.add_out_memlets(
            [preOut], mapExit, tasklet, [shortDims], [shortParams], "lambda a,b: a+b", 0
        )

        mapEntry, mapExit = state.add_map(
            label + "_out", dict(zip(shortParams, shortDims))
        )
        tasklet = state.add_tasklet(label + "_out", {"j0", "j1"}, {"out"}, "out=j0*j1")
        self.add_in_memlets(
            [inputNodes[0], preOut],
            mapEntry,
            tasklet,
            [shortDims, shortDims],
            [shortParams, shortParams],
        )
        self.add_out_memlets(outputList, mapExit, tasklet, [shortDims], [shortParams])

    def visit_LRN(self, node):

        inputList = []
        inputNodes = []
        outputList = []
        state = self.state
        alpha = str(node.get_attr("alpha"))
        beta = str(node.get_attr("beta"))
        bias = str(node.get_attr("bias"))
        depth_radius = str(node.get_attr("depth_radius"))

        for count, inp in enumerate(node.inputs):
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            if count == 0:
                shortDims = []
                shortAccesses = []
                for dim in inp.shape:
                    shortDims.append("0:" + str(dim))
                    shortAccesses.append(str(dim))
                longDims = []
                longDims = shortDims + ["0:" + depth_radius + "*2+1"]
                paddedDims = []
                paddedDims += shortDims
                paddedDims[-1] += "+" + depth_radius + "*2"

        label = _string_builder(node.name)
        outputList = self.create_and_add_output_node(node)
        longParams = ["i0", "i1", "i2", "i3", "i4"]
        shortParams = ["i0", "i1", "i2", "i3"]
        copyParams = ["i0", "i1", "i2", "i3+" + depth_radius]
        normParams = ["i0", "i1", "i2", "i3+i4"]

        paddedShape = []
        paddedShape += shortAccesses
        paddedShape[-1] += "+" + depth_radius
        paddedInput = state.add_transient(
            label + "_paddedInput",
            paddedShape,
            dace.typeclass(_tensortype(node)),
            toplevel=True,
        )
        mapEntry, mapExit = state.add_map(
            label + "_padding", dict(zip(shortParams, shortDims))
        )
        tasklet = state.add_tasklet(label + "_padding", {"j0"}, {"out"}, "out=j0")
        self.add_in_memlets(
            [inputNodes[0]], mapEntry, tasklet, [shortDims], [shortParams]
        )
        self.add_out_memlets(
            [paddedInput], mapExit, tasklet, [paddedDims], [copyParams]
        )

        sqrsum = state.add_transient(
            label + "_Sqrsum", shortAccesses, _tensortype(node), toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            label + "_sqrsum", dict(zip(longParams, longDims))
        )
        tasklet = state.add_tasklet(label + "_sqrsum", {"j0"}, {"out"}, "out=j0*j0")
        self.reinitCR(sqrsum, [shortParams], [shortDims], "0")
        self.add_in_memlets(
            [paddedInput], mapEntry, tasklet, [paddedDims], [normParams]
        )
        self.add_out_memlets(
            [sqrsum], mapExit, tasklet, [shortDims], [shortParams], "lambda a,b: a+b", 0
        )

        mapEntry, mapExit = state.add_map(label, dict(zip(shortParams, shortDims)))
        tasklet = state.add_tasklet(
            _string_builder(node.name),
            {"j0", "j1"},
            {"out"},
            "out = j0/(pow(" + bias + "+" + alpha + "*j1," + beta + "));",
            language=dace.types.Language.CPP,
        )
        self.add_in_memlets(
            (inputNodes + [sqrsum]),
            mapEntry,
            tasklet,
            [shortDims, shortDims],
            [shortParams, shortParams],
        )
        self.add_out_memlets(outputList, mapExit, tasklet, [shortDims], [shortParams])

    def visit_ArgMax(self, node):

        state = self.state
        inputList = []
        inputNodes = []

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode = self.create_and_add_input_node(inp)[0]
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)

                inputAccesses = [[], []]
                inputDims = [[], []]
                inputParams = [[], []]
                for i, dim in enumerate(inp.shape):
                    if i == 0:
                        inputAccesses[1].append(str(dim))
                        inputParams[1].append("i" + str(i))
                        inputDims[1].append("0:" + str(dim))
                    inputAccesses[0].append(str(dim))
                    inputParams[0].append("i" + str(i))
                    inputDims[0].append("0:" + str(dim))

        outputList = self.create_and_add_output_node(node)

        mapLabel = _string_builder(node.name)
        mapEntry, mapExit = state.add_map(
            mapLabel + "_max", dict(zip(inputParams[0], inputDims[0]))
        )
        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(",".join(inputAccesses[1]))
        temp1Node = state.add_transient(
            mapLabel + "_max_tmp", shape, dtype, toplevel=True
        )

        tasklet = state.add_tasklet(mapLabel + "_max", {"j0"}, {"out"}, "out = j0")
        self.reinitCR(temp1Node, [inputParams[1]], [inputDims[1]], "-999999999999")
        self.add_in_memlets(
            [inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputParams[0]]
        )
        self.add_out_memlets(
            [temp1Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: max(a,b)",
            -999999999999,
        )

        mapEntry, mapExit = state.add_map(
            mapLabel + "_arg", dict(zip(inputParams[0], inputDims[0]))
        )
        outputNode = outputList[0]
        tasklet = state.add_tasklet(
            mapLabel + "_map2", {"j0", "j1"}, {"out"}, "if (j0==j1):\n\tout=i1"
        )
        self.add_in_memlets(
            [inputNodes[0], temp1Node], mapEntry, tasklet, inputDims, inputParams
        )
        self.add_out_memlets(
            [outputNode], mapExit, tasklet, [inputDims[1]], [inputParams[1]]
        )

    def visit_Cast(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []
        castType = None

        dtype = node.get_attr("DstT")
        if dtype.as_numpy_dtype == object:
            raise NotImplementedError("Type %s is not a valid numpy type" % str(dtype))
        castType = dace.typeclass(dtype.as_numpy_dtype).ctype

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            params = self.get_default_params(out)
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(
            mapLabel, {"j0"}, {"out"}, "out = " + castType + "(j0)"
        )
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)

    def visit_Print(self, node):
        inputList = []
        inputNodes = []
        outputList = []
        state = self.state
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            params = self.get_default_params(out)
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = _string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))

        ifClause = "if ("
        for param in mapParams:
            ifClause += param + "==1 and "

        ifClause = ifClause[:-4] + "):"
        taskletCode = (
            "out = j0\n" + ifClause + '\n\tprintf("' + inputList[0].label + '")\n'
        )
        taskletCode = 'out = j0\nif(True):\n\tprintf("%f\\n",out)'
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, taskletCode)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Softmax(self, node):

        inputList = []
        inputNodes = []
        state = self.state

        for inp in node.inputs:
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)

        inputDims = [[], []]
        inputParams = [[], []]

        for i, dim in enumerate(inp.shape):
            if i == 0:
                inputParams[1].append("i" + str(i))
                inputDims[1].append("0:" + str(dim))
            inputParams[0].append("i" + str(i))
            inputDims[0].append("0:" + str(dim))

        mapLabel = _string_builder(node.name)
        mapEntry, mapExit = state.add_map(
            mapLabel + "_map1", dict(zip(inputParams[0], inputDims[0]))
        )
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        # 1st map, get maximum in each batchsize dimension
        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(
            str(node.inputs[0].shape.dims[0])
        )
        temp1Node = state.add_transient(
            mapLabel + "_max_tmp", shape, dtype, toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            mapLabel + "_max", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(mapLabel + "_max", {"j0"}, {"out"}, "out = j0")
        self.reinitCR(temp1Node, [inputParams[1]], [inputDims[1]], "-999999999999")
        self.add_in_memlets(
            [inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputParams[0]]
        )
        self.add_out_memlets(
            [temp1Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: max(a,b)",
            -999999999999,
        )

        # 2nd map, calculate the denominator sum
        temp2Node = state.add_transient(
            mapLabel + "_denominator_tmp", shape, dtype, toplevel=True
        )
        mapEntry, mapExit = state.add_map(
            mapLabel + "_denominator", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(
            mapLabel + "_denominator",
            {"j0", "j1"},
            {"out"},
            "out = dace::math::exp(j0-j1);",
            language=dace.types.Language.CPP,
        )
        self.reinitCR(temp2Node, [inputParams[1]], [inputDims[1]], "0")
        inList = [inputNodes[0], temp1Node]
        self.add_in_memlets(inList, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(
            [temp2Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: a+b",
            0,
        )

        # 3rd map, calculate the sofmax
        mapEntry, mapExit = state.add_map(
            mapLabel + "_softmax", dict(zip(mapParams, mapRange))
        )
        tasklet = state.add_tasklet(
            mapLabel + "_softmax",
            {"j0", "j1", "out"},
            {"out"},
            "out = (dace::math::exp(j0-j1))/j2;",
            language=dace.types.Language.CPP,
        )
        inList = [inputList[0], temp1Node, temp2Node]
        paramsList = inputParams + [inputParams[1]]
        dimsList = inputDims + [inputDims[1]]
        self.add_in_memlets(inList, mapEntry, tasklet, dimsList, paramsList)
        self.add_out_memlets(
            outputList, mapExit, tasklet, [inputDims[0]], [inputParams[0]]
        )

    def visit_AddN(self, node):
        inputNodes = []
        inputParams = []
        inputDims = []
        for count, inp in enumerate(node.inputs):
            inpNode, params, dims = self.create_and_add_input_node(inp)
            inputNodes.append(inpNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        outputParams = self.get_default_params(node.outputs[0])
        outputDims = self.get_default_dims(node.outputs[0])
        jays = ["j" + str(index) for index in range(len(inputNodes))]
        tasklet, mapEntry, mapExit = self.state.add_mapped_tasklet(
            _string_builder(node.type),
            dict(zip(inputParams[0], inputDims[0])),
            dict(
                zip(
                    jays,
                    [
                        Memlet.simple(inode, ",".join(params))
                        for inode, params in zip(inputNodes, inputParams)
                    ],
                )
            ),
            "out = " + "+".join(jays),
            dict(out=Memlet.simple(outputList[0], ",".join(outputParams))),
        )
        for inp, dim in zip(inputNodes, inputDims):
            self.state.add_edge(
                inp, None, mapEntry, None, Memlet.simple(inp, ",".join(dim))
            )
        self.state.add_edge(
            mapExit,
            None,
            outputList[0],
            None,
            Memlet.simple(outputList[0], ",".join(outputDims)),
        )

    def add_in_memlets(
        self, inputList, otherNode, tasklet, inputDims, inputParams, identifier="j"
    ):
        """ Convenience function that adds two memlets for each input of the 
            node: external and internal to a given map.
            @param inputList: list of inputNodes (DaCe access node)
            @param otherNode: DaCe node (mostly map_entry)
            @param tasklet: Normally a tasklet node, but it can also be another 
                            mapEntry, for example map in map.
            @param inputDims: List of list of strings dimension of the 
                              respective input. Example:
                              [["0:5","0:7"],["0:2","0:4"]]  
            @param inputParams: List of list of strings params of respective 
                                input. Example: [["i0","i1"],["i2","i3"]]
            @param identifier: This will be used as the base identifier of the
                                input connector to the tasklet. Default is 'j'  
        """
        state = self.state
        connected_nodes = set()
        for i, inp in enumerate(inputList):
            assert isinstance(inputDims[i], list)
            if inp.data not in connected_nodes:
                outerMemlet = Memlet.simple(inp, ",".join(inputDims[i]))
                state.add_edge(inp, None, otherNode, None, outerMemlet)
                connected_nodes.add(inp.data)
            name = identifier + str(i)
            innerMemlet = Memlet.simple(inp, ",".join(inputParams[i]))

            if isinstance(tasklet, (Tasklet, NestedSDFG)):
                state.add_edge(otherNode, None, tasklet, name, innerMemlet)
            else:
                state.add_edge(otherNode, None, tasklet, None, innerMemlet)

    def add_out_memlets(
        self,
        outputList,
        otherNode,
        tasklet,
        outputDims,
        outputParams,
        wcr=None,
        wcr_identity=None,
        identifier="out",
    ):
        """ Convenience function that adds two memlets for each output of the 
            node: external and internal to a given map.
            @param outputList: list of outputNodes (DaCe access node)
            @param otherNode: DaCe node (mostly map_entry)
            @param tasklet: Normally a tasklet node, but it can also be another 
                            mapEntry, for example map in map.
            @param outputDims: List of list of strings dimension of the 
                               respective output. Example:
                               [["0:5","0:7"],["0:2","0:4"]]  
            @param outputParams: List of list of strings params of respective 
                                 output. Example: [["i0","i1"],["i2","i3"]]  
            @param wcr: (optional) Write-conflict resolution function (as 
                        string).
            @param wcr_identity: (optional) Identity element for write-conflict
                                 resolution.
            @param identifier: This is the base identifier for the out connector
                                of the tasklet. Default value is "out". If there are
                                multiple out connectors, each is numbered from zero.
        """

        connected_nodes = set()

        state = self.state
        for i, out in enumerate(outputList):
            assert isinstance(outputDims[i], list)
            if len(outputList) > 1:
                name = identifier + str(i)
            else:
                name = identifier

            if out.data not in connected_nodes:
                outerMemlet = Memlet.simple(
                    out, ",".join(outputDims[i]), wcr_str=wcr, wcr_identity=wcr_identity
                )
                state.add_edge(otherNode, None, out, None, outerMemlet)
                connected_nodes.add(out.data)
            innerMemlet = Memlet.simple(
                out, ",".join(outputParams[i]), wcr_str=wcr, wcr_identity=wcr_identity
            )

            if isinstance(tasklet, (Tasklet, NestedSDFG)):
                state.add_edge(tasklet, name, otherNode, None, innerMemlet)
            else:
                state.add_edge(tasklet, None, otherNode, None, innerMemlet)

    def create_and_add_input_node(self, inp):
        """ Creates a DaCe access node for each input of `inp`, adds it to the 
            state, and returns it.
            If the node already exists, returns the pre-existing node.
            @param inp: tf.Operation
            @return: A 3-tuple of (input DaCe access node, 
                                   list of parameter strings,
                                   list of dimension strings).
        """

        state = self.state
        # Get DaCe name of the operation
        label = _string_builder(inp.name)
        if "?" in str(_tensorshape(inp)):
            raise ValueError("Invalid shape for tensor %s" % label)
        # Try to find node in DaCe graph
        try:
            # If successful, use the existing node
            inputNode = state.find_node(label)
        except (LookupError):
            # Get type and shape of the input tensor
            dtype = dace.typeclass(_tensortype(inp))
            shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(inp)))
            # Create and add array, default is transient, toplevel =True
            inputNode = state.add_transient(
                name=label, shape=shape, dtype=dtype, toplevel=True
            )

        params = self.get_default_params(inp)
        dims = self.get_default_dims(inp)

        return inputNode, params, dims

    def create_and_add_output_node(self, node):
        """ Creates a DaCe access node for each output of `node`, adds it to 
            the state, and returns it.
            If the node already exists, returns the pre-existing node.
            @param node: tf.Operation
            @return: List of DaCe access node.
        """
        outputList = []
        state = self.state
        for count, out in enumerate(node.outputs):
            label = _string_builder(out.name)
            if "?" in str(_tensorshape(out)):
                raise ValueError(
                    "Invalid shape {} for tensor {}".format(_tensorshape(out), label)
                )
            # Iterate over all output nodes
            # Try to find node in DaCe graph
            try:
                # If successful, use the existing node
                outputNode = state.find_node(label)
            except (LookupError):
                # Get type and shape of the tensor
                dtype = dace.typeclass(_tensortype(out))
                shape = dace.properties.ShapeProperty.from_string(
                    str(_tensorshape(out))
                )
                outputNode = state.add_transient(label, shape, dtype, toplevel=True)
            outputList.append(outputNode)
        return outputList

    def reinitCR(self, inp, params, dims, identity):
        """ Adds a reinitialization map to a `reinit` state, setting inputs
            to their initial values. Only used in training mode.
            @param inp: DaCe access node.
            @param params: List of string parameters to `inp`.
            @param dims: List of strings dimensions of `inp`.
            @param identity: Identity value of the CR node (as a string)
        """

        if self.training:
            # Swap current state and reinitState
            self.state, self.reinitState = self.reinitState, self.state
            node = inp
            state = self.state
            dtype = node.desc(self.graph).dtype
            label = node.label

            # Mark node as non-transient as we need to set it from the outside
            # the SDFG.
            node.desc(self.graph).transient = False

            shape = dace.properties.ShapeProperty.from_string(
                str(inp.desc(self.graph).shape)
            )
            # Add input, output and map to reinitState
            inputNode = state.add_array(label, shape, dtype)
            outputNode = state.add_array(label, shape, dtype)
            mapEntry, mapExit = state.add_map(label, dict(zip(params[0], dims[0])))

            # Output is set to identity
            tasklet = state.add_tasklet(label, set(), {"out"}, "out = " + identity)
            state.add_edge(mapEntry, None, tasklet, None, EmptyMemlet())
            self.add_out_memlets([outputNode], mapExit, tasklet, dims, params)
            # Add numpy array with identity value to the reinit dict.
            npArray = np.full(shape, int(identity)).astype(
                node.desc(self.graph).dtype.type
            )
            self.reinitDict.update({label: npArray})
            # Swap state back
            self.reinitState, self.state = self.state, self.reinitState
        else:
            pass

    def inputPadding(
        self, node, inpnode, inp, outputSize, kernelSize, strides, inputDims
    ):
        """ Zero-pads the input to fit the outputSize.
            WARNING: This function assumes the height and width of the output is the
            same (which is reasonable for deep learning).
            @param node: tf.Operation
            @param inpnode: DaCe access node to pad
            @param inp: input node descriptor
            @param outputSize: Output size.
            @param kernelSize: Kernel size.
            @param strides: Strides.
            @param inputDims: List of strings (e.g.["0:N","0:M"]).
            @return: A 2-tuple (output DaCe access node with padded input,
                                list of dimension strings of the padded data).
        """
        state = self.state
        paddingUp = 0
        paddingDown = 0
        label = inpnode.label
        inputSize = inp.shape[1]
        # Calculate padding according to paper
        padding = strides * (outputSize - 1) + kernelSize - inputSize
        # If padding is even (padding is on each side the same)
        if padding % 2 == 0:
            paddingUp = padding // 2
            paddingDown = padding // 2
        # If padding is uneven, we pad more on the bottom and on the right side
        # of an image (matching TensorFlow behavior)
        else:
            paddingUp = padding // 2
            paddingDown = paddingUp + 1

        # Set up the different padding dimensions, accesses and params.
        outputDims = inputDims.copy()
        outputDims[1] = str(paddingUp) + ":" + str(inp.shape[1]) + "+" + str(paddingUp)
        outputDims[2] = str(paddingUp) + ":" + str(inp.shape[2]) + "+" + str(paddingUp)
        outputAccesses = list(map(str, list(inp.shape)))
        outputAccesses[1] += "+" + str(paddingUp) + "+" + str(paddingDown)
        outputAccesses[2] += "+" + str(paddingUp) + "+" + str(paddingDown)
        outputDims = []
        inputParams = []
        for i, dim in enumerate(outputAccesses):
            inputParams.append("i" + str(i))
            outputDims.append("0:" + dim)

        outputParams = inputParams.copy()
        outputParams[1] += "+" + str(paddingUp)
        outputParams[2] += "+" + str(paddingUp)

        # Add the padded input to the graph, set it to zero, and add the map.
        shape = dace.properties.ShapeProperty.from_string(",".join(outputAccesses))
        output = state.add_transient(
            label + "_padded", shape=shape, dtype=inp.dtype, toplevel=True
        )
        output.setzero = True

        mapParams = inputParams
        mapRange = inputDims
        mapLabel = _string_builder(node.type)
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.add_in_memlets([inpnode], mapEntry, tasklet, [inputDims], [inputParams])
        self.add_out_memlets([output], mapExit, tasklet, [outputDims], [outputParams])
        return output, outputDims

    def get_default_params(self, tensor, start=0, identifier="i"):
        """ Returns the default parameters of a tensor starting at `start`,
            e.g., ["i0","i1",...].
            @param tensor: tf.Tensor.
            @param start: Starting position for the iteration.
            @param identifier: The base identifier for the parameters. Default is 'i'
            @return: List of parameters as strings ["i0",i"1",...].
        """
        params = []
        shape = _tensorshape(tensor)
        if shape == 1:
            shape = [1]
        for i, dim in enumerate(shape, start):
            params.append(identifier + str(i))
        return params

    def get_default_dims(self, tensor):
        """ Returns the default dimensions of a tensor e.g., ["0:N","0:M"]
            @param tensor: tf.Tensor.
            @return: List of dimensions as strings ["0:N","0:M"]
        """
        dims = []
        shape = _tensorshape(tensor)
        if shape == 1:
            shape = [1]
        for dim in shape:
            dims.append("0:" + str(dim))
        return dims
