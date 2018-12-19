# new text so i can commit and push


This readme text file is intended as a cursory tensorflow help document taken from various online sources.

* Note: Tensors refers to variables in tensorflow

With tensorflow imported as tf:
-----------------------------------------
tf.placeholder

tf.placeholder(
    dtype,
    shape=None,
    name=None
)
Defined in tensorflow/python/ops/array_ops.py.

Inserts a placeholder for a tensor that will be always fed.

Args:
    dtype: The type of elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any
           shape.
    name: A name for the operation (optional).
Returns:
    A Tensor that may be used as a handle for feeding a value, but not evaluated directly.

Raises:
    RuntimeError: if eager execution is enabled

Eager Compatibility
    Placeholders are not compatible with eager execution.

----------------------------------------

tf.name_scope
Class name_scope
Aliases:
    Class tf.keras.backend.name_scope
    Class tf.name_scope

Defined in tensorflow/python/framework/ops.py.

A context manager for use when defining a Python op.

This context manager validates that the given values are from the same graph, makes that graph the default graph, and
pushes a name scope in that graph (see tf.Graph.name_scope for more details on that).

For example, to define a new Python op called my_op:

def my_op(a, b, c, name=None):
  with tf.name_scope(name, "MyOp", [a, b, c]) as scope:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    c = tf.convert_to_tensor(c, name="c")
    # Define some computation that uses `a`, `b`, and `c`.
    return foo_op(..., name=scope)

__init__
__init__(
    name,
    default_name=None,
    values=None
)
Initialize the context manager.

Args:
    name: The name argument that is passed to the op function.
    default_name: The default name to use if the name argument is None.
    values: The list of Tensor arguments that are passed to the op function.
Properties
name
    Methods
    __enter__
    __enter__()
        Start the scope block.

Returns:
    The scope name.

Raises:
    ValueError: if neither name nor default_name is provided but values are.

__exit__
__exit__(
    type_arg,
    value_arg,
    traceback_arg
)
------------------------------

tf.layers.dense
tf.layers.dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
Defined in tensorflow/python/layers/core.py.

Functional interface for the densely-connected layer.

This layer implements the operation: outputs = activation(inputs * kernel + bias) where activation is the activation
function passed as the activation argument (if not None), kernel is a weights matrix created by the layer, and bias is a
bias vector created by the layer (only if use_bias is True).

Arguments:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix. If None (default), weights are initialized using the
                        default initializer used by tf.get_variable.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the kernel after being updated by an Optimizer
                       (e.g. used to implement norm constraints or value constraints for layer weights). The function
                       must take as input the unprojected variable and must return the projected variable (which must
                       have the same shape). Constraints are not safe to use when doing asynchronous distributed
                       training.
    bias_constraint: An optional projection function to be applied to the bias after being updated by an Optimizer.
    trainable: Boolean, if True also add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES
               (see tf.Variable).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

Returns:
    Output tensor the same shape as inputs except the last dimension is of size units.

Raises:
    ValueError: if eager execution is enabled.

----------------------------------------

tf.keras.initializers.Constant
Class Constant
Inherits From: Initializer

Aliases:
    Class tf.constant_initializer
    Class tf.initializers.constant
    Class tf.keras.initializers.Constant
    Class tf.keras.initializers.constant

Defined in tensorflow/python/ops/init_ops.py.

Initializer that generates tensors with constant values.

The resulting tensor is populated with values of type dtype, as specified by arguments value following the desired shape
of the new tensor (see examples below).

The argument value can be a constant value, or a list of values of type dtype. If value is a list, then the length of
the list must be less than or equal to the number of elements implied by the desired shape of the tensor. In the case
where the total number of elements in value is less than the number of elements required by the tensor shape, the last
element in value will be used to fill the remaining entries. If the total number of elements in value is greater than
the number of elements required by the tensor shape, the initializer will raise a ValueError.

Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy array. All elements of the initialized
           variable will be set to the corresponding value in the value argument.
    dtype: The data type.
    verify_shape: Boolean that enables verification of the shape of value. If True, the initializer will throw an error
                  if the shape of value is not compatible with the shape of the initialized tensor.
Raises:
    TypeError: If the input value is not one of the expected types.
Examples: The following example can be rewritten using a numpy.ndarray instead of the value list, even reshaped, as
          shown in the two commented lines below the value list initialization.

  >>> import numpy as np
  >>> import tensorflow as tf

  >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
  >>> # value = np.array(value)
  >>> # value = value.reshape([2, 4])
  >>> init = tf.constant_initializer(value)

  >>> print('fitting shape:')
  >>> with tf.Session():
  >>>   x = tf.get_variable('x', shape=[2, 4], initializer=init)
  >>>   x.initializer.run()
  >>>   print(x.eval())

  fitting shape:
  [[ 0.  1.  2.  3.]
   [ 4.  5.  6.  7.]]

  >>> print('larger shape:')
  >>> with tf.Session():
  >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init)
  >>>   x.initializer.run()
  >>>   print(x.eval())

  larger shape:
  [[ 0.  1.  2.  3.]
   [ 4.  5.  6.  7.]
   [ 7.  7.  7.  7.]]

  >>> print('smaller shape:')
  >>> with tf.Session():
  >>>   x = tf.get_variable('x', shape=[2, 3], initializer=init)

* <b>`ValueError`</b>: Too many elements provided. Needed at most 6, but received 8

  >>> print('shape verification:')
  >>> init_verify = tf.constant_initializer(value, verify_shape=True)
  >>> with tf.Session():
  >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init_verify)

* <b>`TypeError`</b>: Expected Tensor's shape: (3, 4), got (8,).


__init__
    __init__(
        value=0,
        dtype=tf.float32,
        verify_shape=False
    )
    Initialize self. See help(type(self)) for accurate signature.

Methods
__call__
    __call__(
        shape,
        dtype=None,
        partition_info=None,
        verify_shape=None
    )
    Call self as a function.

from_config
    from_config(
        cls,
        config
    )
    Instantiates an initializer from a configuration dictionary.

Example:
initializer = RandomUniform(-1, 1)
config = initializer.get_config()
initializer = RandomUniform.from_config(config)

Args:
    config: A Python dictionary. It will typically be the output of get_config.
Returns:
    An Initializer instance.

get_config
    get_config()
    Returns the configuration of the initializer as a JSON-serializable dict.

Returns:
    A JSON-serializable Python dict.
-------------------------------------------------
tf.nn
Wrappers for primitive Neural Net (NN) Operations

-------------------------------------------------

tf.math.reduce_sum
Aliases:
    tf.math.reduce_sum
    tf.reduce_sum

tf.math.reduce_sum(
    input_tensor,
    axis=None,
    keepdims=None,
    name=None,
    reduction_indices=None,
    keep_dims=None
)
Defined in tensorflow/python/ops/math_ops.py.

Computes the sum of elements across dimensions of a tensor. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed in a future version. Instructions for updating: keep_dims is
deprecated, use keepdims instead

Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the tensor is reduced by 1
for each entry in axis. If keepdims is true, the reduced dimensions are retained with length 1.

If axis is None, all dimensions are reduced, and a tensor with a single element is returned.

For example:

x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x)  # 6
tf.reduce_sum(x, 0)  # [2, 2, 2]
tf.reduce_sum(x, 1)  # [3, 3]
tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
tf.reduce_sum(x, [0, 1])  # 6

Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If None (the default), reduces all dimensions.
          Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for keepdims.
Returns:
    The reduced tensor, of the same dtype as the input_tensor.

Numpy Compatibility
Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to int64 while tensorflow returns the same
dtype as the input.
-------------------------------------------------

*In digital circuits, one-hot is a group of bits among which the legal combinations of values are only those with a
single high (1) bit and all the others low (0).[1] A similar implementation in which all bits are '1' except one '0' is
sometimes called one-cold.

tf.one_hot
    tf.one_hot(
        indices,
        depth,
        on_value=None,
        off_value=None,
        axis=None,
        dtype=None,
        name=None
    )
Defined in tensorflow/python/ops/array_ops.py.

Returns a one-hot tensor.

The locations represented by indices in indices take value on_value, while all other locations take value off_value.

on_value and off_value must have matching data types. If dtype is also provided, they must be the same data type as
specified by dtype.

If on_value is not provided, it will default to the value 1 with type dtype

If off_value is not provided, it will default to the value 0 with type dtype

If the input indices is rank N, the output will have rank N+1. The new axis is created at dimension axis
(default: the new axis is appended at the end).

If indices is a scalar the output shape will be a vector of length depth

If indices is a vector of length features, the output shape will be:

  features x depth if axis == -1
  depth x features if axis == 0

If indices is a matrix (batch) with shape [batch, features], the output shape will be:

  batch x features x depth if axis == -1
  batch x depth x features if axis == 1
  depth x batch x features if axis == 0

If dtype is not provided, it will attempt to assume the data type of on_value or off_value, if one or both are passed
in. If none of on_value, off_value, or dtype are provided, dtype will default to the value tf.float32.

Note: If a non-numeric data type output is desired (tf.string, tf.bool, etc.), both on_value and off_value must be
      provided to one_hot.

For example:

indices = [0, 1, 2]
depth = 3
tf.one_hot(indices, depth)  # output: [3 x 3]
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

indices = [0, 2, -1, 1]
depth = 3
tf.one_hot(indices, depth,
           on_value=5.0, off_value=0.0,
           axis=-1)  # output: [4 x 3]
# [[5.0, 0.0, 0.0],  # one_hot(0)
#  [0.0, 0.0, 5.0],  # one_hot(2)
#  [0.0, 0.0, 0.0],  # one_hot(-1)
#  [0.0, 5.0, 0.0]]  # one_hot(1)

indices = [[0, 2], [1, -1]]
depth = 3
tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=-1)  # output: [2 x 2 x 3]
# [[[1.0, 0.0, 0.0],   # one_hot(0)
#   [0.0, 0.0, 1.0]],  # one_hot(2)
#  [[0.0, 1.0, 0.0],   # one_hot(1)
#   [0.0, 0.0, 0.0]]]  # one_hot(-1)
Args:
    indices: A Tensor of indices.
    depth: A scalar defining the depth of the one hot dimension.
    on_value: A scalar defining the value to fill in output when indices[j] = i. (default: 1)
    off_value: A scalar defining the value to fill in output when indices[j] != i. (default: 0)
    axis: The axis to fill (default: -1, a new inner-most axis).
    dtype: The data type of the output tensor.
    name: A name for the operation (optional).
Returns:
    output: The one-hot tensor.
Raises:
    TypeError: If dtype of either on_value or off_value don't match dtype
    TypeError: If dtype of on_value and off_value don't match one another

--------------------------------------------------------

tf.nn.softmax
Aliases:
    tf.math.softmax
    tf.nn.softmax

tf.nn.softmax(
    logits,
    axis=None,
    name=None,
    dim=None
)
Defined in tensorflow/python/ops/nn_ops.py.

Computes softmax activations. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed in a future version. Instructions for updating:
dim is deprecated, use axis instead

This function performs the equivalent of

softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
Args:
    logits: A non-empty Tensor. Must be one of the following types: half, float32, float64.
    axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
    name: A name for the operation (optional).
    dim: Deprecated alias for axis.
Returns:
    A Tensor. Has the same type and shape as logits.

Raises:
    InvalidArgumentError: if logits is empty or axis is beyond the last dimension of logits.
-----------------------------------------------------------------

tf.math.log

Aliases:
    tf.log
    tf.math.log
tf.math.log(
    x,
    name=None
)
Defined in generated file: tensorflow/python/ops/gen_math_ops.py.

Computes natural logarithm of x element-wise.

I.e., .

Args:
    x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
       name: A name for the operation (optional).
Returns:
    A Tensor. Has the same type as x

-----------------------------------------------------------------

tf.math.reduce_mean
Aliases:
    tf.math.reduce_mean
    tf.reduce_mean

tf.math.reduce_mean(
    input_tensor,
    axis=None,
    keepdims=None,
    name=None,
    reduction_indices=None,
    keep_dims=None
)
Defined in tensorflow/python/ops/math_ops.py.

Computes the mean of elements across dimensions of a tensor. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed in a future version. Instructions for updating: keep_dims is
deprecated, use keepdims instead

Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the tensor is reduced by 1
for each entry in axis. If keepdims is true, the reduced dimensions are retained with length 1.

If axis is None, all dimensions are reduced, and a tensor with a single element is returned.

For example:

x = tf.constant([[1., 1.], [2., 2.]])
tf.reduce_mean(x)  # 1.5
tf.reduce_mean(x, 0)  # [1.5, 1.5]
tf.reduce_mean(x, 1)  # [1.,  2.]

Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range
          [-rank(input_tensor), rank(input_tensor)).
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for keepdims.
Returns:
    The reduced tensor.

Numpy Compatibility
Equivalent to np.mean

Please note that np.mean has a dtype parameter that could be used to specify the output type. By default this is
dtype=float64. On the other hand, tf.reduce_mean has an aggressive type inference from input_tensor, for example:

x = tf.constant([1, 0, 1, 0])
tf.reduce_mean(x)  # 0
y = tf.constant([1., 0., 1., 0.])
tf.reduce_mean(y)  # 0.5
-----------------------------------------------------------------
https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
tf.train.AdamOptimizer
Class AdamOptimizer
Inherits From: Optimizer

Defined in tensorflow/python/training/adam.py.

Optimizer that implements the Adam algorithm.

See Kingma et al., 2014 (pdf).

__init__
__init__(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
)
Construct a new Adam optimizer.

The update rule for variable with gradient g uses an optimization described at the end of section2 of the paper:

The default value of 1e-8 for epsilon might not be a good default in general. For example, when training an Inception
network on ImageNet a current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the formulation just before
Section 2.1 of the Kingma and Ba paper rather than the formulation in Algorithm 1, the "epsilon" referred to here is
"epsilon hat" in the paper.

The sparse implementation of this algorithm (used when the gradient is an IndexedSlices object, typically because of
tf.gather or an embedding lookup in the forward pass) does apply momentum to variable slices even if they were not used
 in the forward pass (meaning they have a gradient equal to zero). Momentum decay (beta1) is also applied to the entire
 momentum accumulator. This means that the sparse behavior is equivalent to the dense behavior (in contrast to some
 momentum implementations which ignore momentum unless a variable slice was actually used).

Args:
    learning_rate: A Tensor or a floating point value. The learning rate.
    beta1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
    beta2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
    epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper
             (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
    use_locking: If True use locks for update operations.
    name: Optional name for the operations created when applying gradients. Defaults to "Adam".
Eager Compatibility
When eager execution is enabled, learning_rate, beta1, beta2, and epsilon can each be a callable that takes no arguments
and returns the actual value to use. This can be useful for changing these values across different invocations of
optimizer functions.

Methods
apply_gradients
apply_gradients(
    grads_and_vars,
    global_step=None,
    name=None
)
Apply gradients to variables.

This is the second part of minimize(). It returns an Operation that applies gradients.

Args:
    grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients().
    global_step: Optional Variable to increment by one after the variables have been updated.
    name: Optional name for the returned operation. Default to the name passed to the Optimizer constructor.
Returns:
    An Operation that applies the specified gradients. If global_step was not None, that operation also increments
global_step.

Raises:
    TypeError: If grads_and_vars is malformed.
    ValueError: If none of the variables have gradients.
    RuntimeError: If you should use _distributed_apply() instead.

compute_gradients
compute_gradients(
    loss,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    grad_loss=None
)
Compute gradients of loss for the variables in var_list.

This is the first part of minimize(). It returns a list of (gradient, variable) pairs where "gradient" is the gradient
for "variable". Note that "gradient" can be a Tensor, an IndexedSlices, or None if there is no gradient for the given
variable.

Args:
    loss: A Tensor containing the value to minimize or a callable taking no arguments which returns the value to
          minimize. When eager execution is enabled it must be a callable.
    var_list: Optional list or tuple of tf.Variable to update to minimize loss. Defaults to the list of variables
              collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.
    gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
    aggregation_method: Specifies the method used to combine gradient terms. Valid values are defined in the class
                        AggregationMethod.
    colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op.
    grad_loss: Optional. A Tensor holding the gradient computed for loss.
Returns:
    A list of (gradient, variable) pairs. Variable is always present, but gradient can be None.

Raises:
    TypeError: If var_list contains anything else than Variable objects.
    ValueError: If some arguments are invalid.
    RuntimeError: If called with eager execution enabled and loss is not callable.
Eager Compatibility
    When eager execution is enabled, gate_gradients, aggregation_method, and colocate_gradients_with_ops are ignored.

get_name
get_name()

get_slot
get_slot(
    var,
    name
)
Return a slot named name created for var by the Optimizer.

Some Optimizer subclasses use additional variables. For example Momentum and Adagrad use variables to accumulate
updates. This method gives access to these Variable objects if for some reason you need them.

Use get_slot_names() to get the list of slot names created by the Optimizer.

Args:
var: A variable passed to minimize() or apply_gradients().
name: A string.
Returns:
The Variable for the slot if it was created, None otherwise.

get_slot_names
get_slot_names()
Return a list of the names of slots created by the Optimizer.

See get_slot().

Returns:
A list of strings.

minimize
minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)
Add operations to minimize loss by updating var_list.

This method simply combines calls compute_gradients() and apply_gradients(). If you want to process the gradient before
applying them call compute_gradients() and apply_gradients() explicitly instead of using this function.

Args:
    loss: A Tensor containing the value to minimize.
    global_step: Optional Variable to increment by one after the variables have been updated.
    var_list: Optional list or tuple of Variable objects to update to minimize loss. Defaults to the list of variables
              collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.
    gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
    aggregation_method: Specifies the method used to combine gradient terms. Valid values are defined in the class
                        AggregationMethod.
    colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op.
    name: Optional name for the returned operation.
    grad_loss: Optional. A Tensor holding the gradient computed for loss.
Returns:
    An Operation that updates the variables in var_list. If global_step was not None, that operation also increments
    global_step.

Raises:
    ValueError: If some of the variables are not Variable objects.
Eager Compatibility
    When eager execution is enabled, loss should be a Python function that takes no arguments and computes the value to
    be minimized. Minimization (and gradient computation) is done with respect to the elements of var_list if not None,
    else with respect to any trainable variables created during the execution of the loss function. gate_gradients,
    aggregation_method, colocate_gradients_with_ops and grad_loss are ignored when eager execution is enabled.

variables
variables()
A list of variables which encode the current state of Optimizer.

Includes slot variables and additional global variables created by the optimizer in the current default graph.

Returns:
    A list of variables.

-----------------------------------------------------------------

tf.Session
Class Session
Defined in tensorflow/python/client/session.py.

A class for running TensorFlow operations.

A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.
For example:

# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print(sess.run(c))
A session may own resources, such as tf.Variable, tf.QueueBase, and tf.ReaderBase. It is important to release these
resources when they are no longer required. To do this, either invoke the tf.Session.close method on the session, or use
 the session as a context manager. The following two examples are equivalent:

# Using the `close()` method.
sess = tf.Session()
sess.run(...)
sess.close()

# Using the context manager.
with tf.Session() as sess:
  sess.run(...)
The ConfigProto protocol buffer exposes various configuration options for a session. For example, to create a session
that uses soft constraints for device placement, and log the resulting placement decisions, create a session as follows:

# Launch the graph in a session that allows soft device placement and
# logs the placement decisions.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))

__init__
__init__(
    target='',
    graph=None,
    config=None
)
Creates a new TensorFlow session.

If no graph argument is specified when constructing the session, the default graph will be launched in the session.
If you are using more than one graph (created with tf.Graph() in the same process, you will have to use different
sessions for each graph, but each graph can be used in multiple sessions. In this case, it is often clearer to pass the
graph to be launched explicitly to the session constructor.

Args:
    target: (Optional.) The execution engine to connect to. Defaults to using an in-process engine.
            See Distributed TensorFlow for more examples.
    graph: (Optional.) The Graph to be launched (described above).
    config: (Optional.) A ConfigProto protocol buffer with configuration options for the session.
Properties
    graph
        The graph that was launched in this session.

    graph_def
        A serializable version of the underlying TensorFlow graph.

        Returns:
            A graph_pb2.GraphDef proto containing nodes for all of the Operations in the underlying TensorFlow graph.

    sess_str
        The TensorFlow process to which this session will connect.

Methods
__enter__
__enter__()

__exit__
__exit__(
    exec_type,
    exec_value,
    exec_tb
)

as_default
as_default()
Returns a context manager that makes this object the default session.

Use with the with keyword to specify that calls to tf.Operation.run or tf.Tensor.eval should be executed in this session.

c = tf.constant(..)
sess = tf.Session()

with sess.as_default():
  assert tf.get_default_session() is sess
  print(c.eval())
To get the current default session, use tf.get_default_session.

N.B. The as_default context manager does not close the session when you exit the context, and you must close the session
explicitly.

c = tf.constant(...)
sess = tf.Session()
with sess.as_default():
  print(c.eval())
# ...
with sess.as_default():
  print(c.eval())

sess.close()

Alternatively, you can use with tf.Session(): to create a session that is automatically closed on exiting the context,
including when an uncaught exception is raised.

N.B. The default session is a property of the current thread. If you create a new thread, and wish to use the default
session in that thread, you must explicitly add a with sess.as_default(): in that thread's function.

N.B. Entering a with sess.as_default(): block does not affect the current default graph. If you are using multiple
graphs, and sess.graph is different from the value of tf.get_default_graph, you must explicitly enter a with
sess.graph.as_default(): block to make sess.graph the default graph.

Returns:
    A context manager using this session as the default session.

close
close()
Closes this session.

Calling this method frees all resources associated with the session.

Raises:
    tf.errors.OpError: Or one of its subclasses if an error occurs while closing the TensorFlow session.

list_devices
list_devices()
Lists available devices in this session.

devices = sess.list_devices()
for d in devices:
  print(d.name)
Each element in the list has the following properties: - name: A string with the full name of the device.
ex: /job:worker/replica:0/task:3/device:CPU:0 - device_type: The type of the device (e.g. CPU, GPU, TPU.) - memory_limit:
The maximum amount of memory available on the device.
Note: depending on the device, it is possible the usable memory could be substantially less.

Raises:
tf.errors.OpError: If it encounters an error (e.g. session is in an invalid state, or network errors occur).
Returns:
A list of devices in the session.

make_callable
make_callable(
    fetches,
    feed_list=None,
    accept_options=False
)
Returns a Python callable that runs a particular step.

The returned callable will take len(feed_list) arguments whose types must be compatible feed values for the respective
elements of feed_list. For example, if element i of feed_list is a tf.Tensor, the ith argument to the returned callable
must be a numpy ndarray (or something convertible to an ndarray) with matching element type and shape.
See tf.Session.run for details of the allowable feed key and value types.

The returned callable will have the same return type as tf.Session.run(fetches, ...). For example, if fetches is a
tf.Tensor, the callable will return a numpy ndarray; if fetches is a tf.Operation, it will return None.

Args:
    fetches: A value or list of values to fetch. See tf.Session.run for details of the allowable fetch types.
    feed_list: (Optional.) A list of feed_dict keys. See tf.Session.run for details of the allowable feed key types.
    accept_options: (Optional.) If True, the returned Callable will be able to accept tf.RunOptions and tf.RunMetadata
                    as optional keyword arguments options and run_metadata, respectively, with the same syntax and
                    semantics as tf.Session.run, which is useful for certain use cases (profiling and debugging) but
                    will result in measurable slowdown of the Callable's performance. Default: False.
Returns:
    A function that when called will execute the step defined by feed_list and fetches in this session.

Raises:
    TypeError: If fetches or feed_list cannot be interpreted as arguments to tf.Session.run.

partial_run
partial_run(
    handle,
    fetches,
    feed_dict=None
)
Continues the execution with more feeds and fetches.

This is EXPERIMENTAL and subject to change.

To use partial execution, a user first calls partial_run_setup() and then a sequence of partial_run(). partial_run_setup
 specifies the list of feeds and fetches that will be used in the subsequent partial_run calls.

The optional feed_dict argument allows the caller to override the value of tensors in the graph.
See run() for more information.

Below is a simple example:

a = array_ops.placeholder(dtypes.float32, shape=[])
b = array_ops.placeholder(dtypes.float32, shape=[])
c = array_ops.placeholder(dtypes.float32, shape=[])
r1 = math_ops.add(a, b)
r2 = math_ops.multiply(r1, c)

h = sess.partial_run_setup([r1, r2], [a, b, c])
res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
res = sess.partial_run(h, r2, feed_dict={c: res})
Args:
    handle: A handle for a sequence of partial runs.
    fetches: A single graph element, a list of graph elements, or a dictionary whose values are graph elements or lists
             of graph elements (see documentation for run).
    feed_dict: A dictionary that maps graph elements to values (described above).
Returns:
    Either a single value if fetches is a single graph element, or a list of values if fetches is a list, or a
    dictionary with the same keys as fetches if that is a dictionary (see documentation for run).

Raises:
    tf.errors.OpError: Or one of its subclasses on error.

partial_run_setup
partial_run_setup(
    fetches,
    feeds=None
)
Sets up a graph with feeds and fetches for partial run.

This is EXPERIMENTAL and subject to change.

Note that contrary to run, feeds only specifies the graph elements.
The tensors will be supplied by the subsequent partial_run calls.

Args:
    fetches: A single graph element, or a list of graph elements.
    feeds: A single graph element, or a list of graph elements.
Returns:
    A handle for partial run.

Raises:
    RuntimeError: If this Session is in an invalid state (e.g. has been closed).
    TypeError: If fetches or feed_dict keys are of an inappropriate type.
    tf.errors.OpError: Or one of its subclasses if a TensorFlow error happens.

reset
@staticmethod
reset(
    target,
    containers=None,
    config=None
)
Resets resource containers on target, and close all connected sessions.

A resource container is distributed across all workers in the same cluster as target. When a resource container on
target is reset, resources associated with that container will be cleared. In particular, all Variables in the container
 will become undefined: they lose their values and shapes.

NOTE: (i) reset() is currently only implemented for distributed sessions. (ii) Any sessions on the master named by
target will be closed.

If no resource containers are provided, all containers are reset.

Args:
    target: The execution engine to connect to.
    containers: A list of resource container name strings, or None if all of all the containers are to be reset.
    config: (Optional.) Protocol buffer with configuration options.
Raises:
    tf.errors.OpError: Or one of its subclasses if an error occurs while resetting containers.

run
run(
    fetches,
    feed_dict=None,
    options=None,
    run_metadata=None
)
Runs operations and evaluates tensors in fetches.

This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every
Operation and evaluate every Tensor in fetches, substituting the values in feed_dict for the corresponding input values.

The fetches argument may be a single graph element, or an arbitrarily nested list, tuple, namedtuple, dict, or
OrderedDict containing graph elements at its leaves. A graph element can be one of the following types:

An tf.Operation. The corresponding fetched value will be None.
A tf.Tensor. The corresponding fetched value will be a numpy ndarray containing the value of that tensor.
A tf.SparseTensor. The corresponding fetched value will be a tf.SparseTensorValue containing the value of that
    sparse tensor.
A get_tensor_handle op. The corresponding fetched value will be a numpy ndarray containing the handle of that tensor.
A string which is the name of a tensor or operation in the graph.
The value returned by run() has the same shape as the fetches argument, where the leaves are replaced by the
    corresponding values returned by TensorFlow.

Example:

   a = tf.constant([10, 20])
   b = tf.constant([1.0, 2.0])
   # 'fetches' can be a singleton
   v = session.run(a)
   # v is the numpy array [10, 20]
   # 'fetches' can be a list.
   v = session.run([a, b])
   # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
   # 1-D array [1.0, 2.0]
   # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
   MyData = collections.namedtuple('MyData', ['a', 'b'])
   v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
   # v is a dict with
   # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
   # 'b' (the numpy array [1.0, 2.0])
   # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
   # [10, 20].

The optional feed_dict argument allows the caller to override the value of tensors in the graph.
Each key in feed_dict can be one of the following types:

If the key is a tf.Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the
same dtype as that tensor. Additionally, if the key is a tf.placeholder, the shape of the value will be checked for
compatibility with the placeholder.
If the key is a tf.SparseTensor, the value should be a tf.SparseTensorValue.
If the key is a nested tuple of Tensors or SparseTensors, the value should be a nested tuple with the same structure
that maps to their corresponding values as above.
Each value in feed_dict must be convertible to a numpy array of the dtype of the corresponding key.

The optional options argument expects a [RunOptions] proto. The options allow controlling the behavior of this
particular step (e.g. turning tracing on).

The optional run_metadata argument expects a [RunMetadata] proto. When appropriate, the non-Tensor output of this step
will be collected there. For example, when users turn on tracing in options, the profiled info will be collected into
this argument and passed back.

Args:
    fetches: A single graph element, a list of graph elements, or a dictionary whose values are graph elements or lists
             of graph elements (described above).
    feed_dict: A dictionary that maps graph elements to values (described above).
    options: A [RunOptions] protocol buffer
    run_metadata: A [RunMetadata] protocol buffer
Returns:
    Either a single value if fetches is a single graph element, or a list of values if fetches is a list, or a
    dictionary with the same keys as fetches if that is a dictionary (described above). Order in which fetches
    operations are evaluated inside the call is undefined.

Raises:
    RuntimeError: If this Session is in an invalid state (e.g. has been closed).
    TypeError: If fetches or feed_dict keys are of an inappropriate type.
    ValueError: If fetches or feed_dict keys are invalid or refer to a Tensor that doesn't exist.

-----------------------------------------------------------------

numpy.ravel¶
numpy.ravel(a, order='C')[source]
Return a contiguous flattened array.

A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

As of NumPy 1.10, the returned array will have the same type as the input array. (for example, a masked array will be
returned for a masked array input)

Parameters:
a : array_like
Input array. The elements in a are read in the order specified by order, and packed as a 1-D array.

order : {‘C’,’F’, ‘A’, ‘K’}, optional
The elements of a are read using this index order. ‘C’ means to index the elements in row-major, C-style order, with the
last axis index changing fastest, back to the first axis index changing slowest. ‘F’ means to index the elements in
column-major, Fortran-style order, with the first index changing fastest, and the last index changing slowest. Note that
 the ‘C’ and ‘F’ options take no account of the memory layout of the underlying array, and only refer to the order of
 axis indexing. ‘A’ means to read the elements in Fortran-like index order if a is Fortran contiguous in memory, C-like
 order otherwise. ‘K’ means to read the elements in the order they occur in memory, except for reversing the data when
 strides are negative. By default, ‘C’ index order is used.

Returns:
y : array_like
y is an array of the same subtype as a, with shape (a.size,). Note that matrices are special cased for backward
compatibility, if a is a matrix, then y is a 1-D ndarray.

See also
ndarray.flat
1-D iterator over an array.
ndarray.flatten
1-D array copy of the elements of an array in row-major order.
ndarray.reshape
Change the shape of an array without changing its data.
Notes

In row-major, C-style order, in two dimensions, the row index varies the slowest, and the column index the quickest.
This can be generalized to multiple dimensions, where row-major order implies that the index along the first axis varies
 slowest, and the index along the last quickest. The opposite holds for column-major, Fortran-style index ordering.

When a view is desired in as many cases as possible, arr.reshape(-1) may be preferable.

Examples

It is equivalent to reshape(-1, order=order).

>>>
>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> print(np.ravel(x))
[1 2 3 4 5 6]
>>>
>>> print(x.reshape(-1))
[1 2 3 4 5 6]
>>>
>>> print(np.ravel(x, order='F'))
[1 4 2 5 3 6]
When order is ‘A’, it will preserve the array’s ‘C’ or ‘F’ ordering:

>>>
>>> print(np.ravel(x.T))
[1 4 2 5 3 6]
>>> print(np.ravel(x.T, order='A'))
[1 2 3 4 5 6]
When order is ‘K’, it will preserve orderings that are neither ‘C’ nor ‘F’, but won’t reverse axes:

>>>
>>> a = np.arange(3)[::-1]; a
array([2, 1, 0])
>>> a.ravel(order='C')
array([2, 1, 0])
>>> a.ravel(order='K')
array([2, 1, 0])
>>>
>>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
array([[[ 0,  2,  4],
        [ 1,  3,  5]],
       [[ 6,  8, 10],
        [ 7,  9, 11]]])
>>> a.ravel(order='C')
array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
>>> a.ravel(order='K')
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

-----------------------------------------------------------------

numpy.reshape¶
numpy.reshape(a, newshape, order='C')[source]
Gives a new shape to an array without changing its data.

Parameters:
a : array_like
Array to be reshaped.

newshape : int or tuple of ints
The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that
length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining
dimensions.

order : {‘C’, ‘F’, ‘A’}, optional
Read the elements of a using this index order, and place the elements into the reshaped array using this index order.
‘C’ means to read / write the elements using C-like index order, with the last axis index changing fastest, back to the
first axis index changing slowest. ‘F’ means to read / write the elements using Fortran-like index order, with the first
 index changing fastest, and the last index changing slowest. Note that the ‘C’ and ‘F’ options take no account of the
 memory layout of the underlying array, and only refer to the order of indexing. ‘A’ means to read / write the elements
  in Fortran-like index order if a is Fortran contiguous in memory, C-like order otherwise.

Returns:
reshaped_array : ndarray
This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory
layout (C- or Fortran- contiguous) of the returned array.

See also
ndarray.reshape
Equivalent method.
Notes

It is not always possible to change the shape of an array without copying the data. If you want an error to be raised
when the data is copied, you should assign the new shape to the shape attribute of the array:

>>>
>>> a = np.zeros((10, 2))
# A transpose makes the array non-contiguous
>>> b = a.T
# Taking a view makes it possible to modify the shape without modifying
# the initial object.
>>> c = b.view()
>>> c.shape = (20)
AttributeError: incompatible shape for a non-contiguous array
The order keyword gives the index ordering both for fetching the values from a, and then placing the values into the
output array. For example, let’s say you have an array:

>>>
>>> a = np.arange(6).reshape((3, 2))
>>> a
array([[0, 1],
       [2, 3],
       [4, 5]])
You can think of reshaping as first raveling the array (using the given index order), then inserting the elements from
the raveled array into the new array using the same kind of index ordering as was used for the raveling.

>>>
>>> np.reshape(a, (2, 3)) # C-like index ordering
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
array([[0, 4, 3],
       [2, 1, 5]])
>>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
array([[0, 4, 3],
       [2, 1, 5]])
Examples

>>>
>>> a = np.array([[1,2,3], [4,5,6]])
>>> np.reshape(a, 6)
array([1, 2, 3, 4, 5, 6])
>>> np.reshape(a, 6, order='F')
array([1, 4, 2, 5, 3, 6])
>>>
>>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
array([[1, 2],
       [3, 4],
       [5, 6]])

-----------------------------------------------------------------

heapq — Heap queue algorithm
This module provides an implementation of the heap queue algorithm, also known as the priority queue algorithm.

Heaps are arrays for which heap[k] <= heap[2*k+1] and heap[k] <= heap[2*k+2] for all k, counting elements from zero. For
    the sake of comparison, non-existing elements are considered to be infinite.
The interesting property of a heap is that heap[0] is always its smallest element.

The API below differs from textbook heap algorithms in two aspects:
    (a) We use zero-based indexing. This makes the relationship between the index for a node and the indexes for its
        children slightly less obvious, but is more suitable since Python uses zero-based indexing.
    (b) Our pop method returns the smallest item, not the largest (called a “min heap” in textbooks; a “max heap” is
        more common in texts because of its suitability for in-place sorting).

These two make it possible to view the heap as a regular Python list without surprises:
    heap[0] is the smallest item, and heap.sort() maintains the heap invariant!

To create a heap, use a list initialized to [], or you can transform a populated list into a heap via function heapify().

The following functions are provided:

heapq.heappush(heap, item)
    Push the value item onto the heap, maintaining the heap invariant.
heapq.heappop(heap)
    Pop and return the smallest item from the heap, maintaining the heap invariant.
    If the heap is empty, IndexError is raised.
heapq.heappushpop(heap, item)
    Push item on the heap, then pop and return the smallest item from the heap.
    The combined action runs more efficiently than heappush() followed by a separate call to heappop().
heapq.heapify(x)
    Transform list x into a heap, in-place, in linear time.
heapq.heapreplace(heap, item)
    Pop and return the smallest item from the heap, and also push the new item. The heap size doesn’t change.
    If the heap is empty, IndexError is raised. This is more efficient than heappop() followed by heappush(), and can be
        more appropriate when using a fixed-size heap.
    Note that the value returned may be larger than item! That constrains reasonable uses of this routine unless written
        as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)

Example of use:

>>> from heapq import heappush, heappop
>>> heap = []
>>> data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
>>> for item in data:
...     heappush(heap, item)
...
>>> ordered = []
>>> while heap:
...     ordered.append(heappop(heap))
...
>>> ordered
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> data.sort()
>>> data == ordered
True

Using a heap to insert items at the correct place in a priority queue:

>>> heap = []
>>> data = [(1, 'J'), (4, 'N'), (3, 'H'), (2, 'O')]
>>> for item in data:
...     heappush(heap, item)
...
>>> while heap:
...     print(heappop(heap)[1])
J
O
H
N

The module also offers three general purpose functions based on heaps.

heapq.merge(*iterables)
    Merge multiple sorted inputs into a single sorted output (for example, merge timestamped entries from multiple
        log files). Returns an iterator over the sorted values.
    Similar to sorted(itertools.chain(*iterables)) but returns an iterable, does not pull the data into memory all at
        once, and assumes that each of the input streams is already sorted (smallest to largest).

heapq.nlargest(n, iterable[, key])
    Return a list with the n largest elements from the dataset defined by iterable. key, if provided, specifies a
        function of one argument that is used to extract a comparison key from each element in the iterable:
        key=str.lower Equivalent to: sorted(iterable, key=key, reverse=True)[:n]
heapq.nsmallest(n, iterable[, key])
    Return a list with the n smallest elements from the dataset defined by iterable. key, if provided, specifies a
    function of one argument that is used to extract a comparison key from each element in the iterable: key=str.lower
    Equivalent to: sorted(iterable, key=key)[:n]

The latter two functions perform best for smaller values of n. For larger values, it is more efficient to use the
    sorted() function. Also, when n==1, it is more efficient to use the builtin min() and max() functions.

Theory
(This explanation is due to François Pinard. The Python code for this module was contributed by Kevin O’Connor.)

Heaps are arrays for which a[k] <= a[2*k+1] and a[k] <= a[2*k+2] for all k, counting elements from 0. For the sake of
    comparison, non-existing elements are considered to be infinite. The interesting property of a heap is that a[0] is
    always its smallest element.

The strange invariant above is meant to be an efficient memory representation for a tournament.
    The numbers below are k, not a[k]:

                               0

              1                                 2

      3               4                5               6

  7       8       9       10      11      12      13      14

15 16   17 18   19 20   21 22   23 24   25 26   27 28   29 30

In the tree above, each cell k is topping 2*k+1 and 2*k+2. In an usual binary tournament we see in sports, each cell is
    the winner over the two cells it tops, and we can trace the winner down the tree to see all opponents s/he had.
    However, in many computer applications of such tournaments, we do not need to trace the history of a winner. To be
    more memory efficient, when a winner is promoted, we try to replace it by something else at a lower level, and the
    rule becomes that a cell and the two cells it tops contain three different items, but the top cell “wins” over the
    two topped cells.

If this heap invariant is protected at all time, index 0 is clearly the overall winner. The simplest algorithmic way to
    remove it and find the “next” winner is to move some loser (let’s say cell 30 in the diagram above) into the 0
    position, and then percolate this new 0 down the tree, exchanging values, until the invariant is re-established.
    This is clearly logarithmic on the total number of items in the tree. By iterating over all items, you get an
    O(n log n) sort.

A nice feature of this sort is that you can efficiently insert new items while the sort is going on, provided that the
    inserted items are not “better” than the last 0’th element you extracted. This is especially useful in simulation
    contexts, where the tree holds all incoming events, and the “win” condition means the smallest scheduled time. When
    an event schedule other events for execution, they are scheduled into the future, so they can easily go into the
    heap. So, a heap is a good structure for implementing schedulers (this is what I used for my MIDI sequencer :-).

Various structures for implementing schedulers have been extensively studied, and heaps are good for this, as they are
    reasonably speedy, the speed is almost constant, and the worst case is not much different than the average case.
    However, there are other representations which are more efficient overall, yet the worst cases might be terrible.

Heaps are also very useful in big disk sorts. You most probably all know that a big sort implies producing “runs”
    (which are pre-sorted sequences, which size is usually related to the amount of CPU memory),
    followed by a merging passes for these runs, which merging is often very cleverly organised [1]. It is very
    important that the initial sort produces the longest runs possible. Tournaments are a good way to that. If, using
    all the memory available to hold a tournament, you replace and percolate items that happen to fit the current run,
    you’ll produce runs which are twice the size of the memory for random input, and much better for input fuzzily
    ordered.

Moreover, if you output the 0’th item on disk and get an input which may not fit in the current tournament (because the
    value “wins” over the last output value), it cannot fit in the heap, so the size of the heap decreases. The freed
    memory could be cleverly reused immediately for progressively building a second heap, which grows at exactly the
    same rate the first heap is melting. When the first heap completely vanishes, you switch heaps and start a new run.
    Clever and quite effective!

In a word, heaps are useful memory structures to know. I use them in a few applications, and I think it is good to
    keep a ‘heap’ module around. :-)

Footnotes

[1]	The disk balancing algorithms which are current, nowadays, are more annoying than clever, and this is a consequence
    of the seeking capabilities of the disks. On devices which cannot seek, like big tape drives, the story was quite
    different, and one had to be very clever to ensure (far in advance) that each tape movement will be the most
    effective possible (that is, will best participate at “progressing” the merge). Some tapes were even able to read
    backwards, and this was also used to avoid the rewinding time. Believe me, real good tape sorts were quite
    spectacular to watch! From all times, sorting has always been a Great Art! :-)

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

-----------------------------------------------------------------

