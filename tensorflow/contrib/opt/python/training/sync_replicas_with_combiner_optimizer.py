# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Wrapper optimizer for Model Average."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook

GLOBAL_VARIABLE_NAME = "global_center_variable"

class CombinerCustomGetter(object):
  """Custom_getter class is used to do.

  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device. Besides, use 'tf.get_variable' instead of 'tf.Variable' to
    use this custom getter.

  For example,
  ma_custom_getter = ModelAverageCustomGetter(worker_device)
  with tf.device(
    tf.train.replica_device_setter(
      worker_device=worker_device,
      ps_device="/job:ps/cpu:0",is_combiner
      cluster=cluster)),
    tf.variable_scope('',custom_getter=ma_custom_getter):
    hid_w = tf.get_variable(
      initializer=tf.truncated_normal(
          [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
          stddev=1.0 / IMAGE_PIXELS),
      name="hid_w")
    hid_b = tf.get_variable(initializer=tf.zeros([FLAGS.hidden_units]),
                            name="hid_b")
  """

  def __init__(self, combiner_device, combiner_field):
    """Create a new `CombinerCustomGetter`.

    Args:
      combiner_device: String.  Name of the `combiner` job.
    """
    self._combiner_device = combiner_device
    self._combiner_field = combiner_field
    self._combiner_2_global = {}

  def __call__(self, getter, name, reuse, trainable, collections, *args, **kwargs):
    print(self._combiner_device)
    if trainable:
      with ops.device(self._combiner_device):
        combiner_var = getter(
            name="%s/%s" % (self._combiner_field, name),
            reuse=variable_scope.AUTO_REUSE,
            trainable=True,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES],
            *args,
            **kwargs)

      global_variable = variable_scope.variable(
          name="%s/%s" % (GLOBAL_VARIABLE_NAME, name),
          initial_value=combiner_var.initialized_value(),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES])

      print(combiner_var, global_variable)
      self._combiner_2_global[combiner_var] = global_variable
      return combiner_var
    else:
      kwargs['trainable'] = trainable
      kwargs['collections'] = collections
      if ops.GraphKeys.LOCAL_VARIABLES in collections:
        with ops.device(self._combiner_device):
          return getter(name, *args, **kwargs)
      else:
        return getter(name, *args, **kwargs)


class SyncReplicasWithCombinerOptimizer(optimizer.Optimizer):
  """Wrapper optimizer that implements the Model Average algorithm.

  This is a sync optimizer. During the training, each worker will update
  the local variables and maintains its own local_step, which starts from 0
  and is incremented by 1 after each update of local variables. Whenever the
  interval_steps divides the local step, the local variables from all the
  workers will be averaged and assigned to global center variables. Then the
  local variables will be assigned by global center variables.
  """

  def __init__(self,
               opt,
               num_worker,
               num_combiner,
               is_controller,
               is_chief,
               combiner_custom_getter,
               use_locking=True,
               name="sync_replicas_with_combiner"):

    """
    Args:
      opt: The actual optimizer that will be used to update local variables
      num_worker: The number of workers
      num_combiner: The number of combiner
      is_controller: whether combiner controller worker
      is_chief: whether chief worker
      combiner_custom_getter: ModelAverageCustomGetter
      use_locking: If True use locks for update operations
      name: string. Optional name of the returned operation
    """
    super(SyncReplicasWithCombinerOptimizer, self).__init__(use_locking, name)
    self._opt = opt
    self._num_worker = num_worker
    self._num_combiner = num_combiner
    self._is_controller = is_controller
    self._is_chief = is_chief
    self._combiner_2_global = combiner_custom_getter._combiner_2_global  # pylint:disable=protected-access
    self._combiner_field = combiner_custom_getter._combiner_field
    self._combiner_device = combiner_custom_getter._combiner_device
    self._accumulator_list = []
    self._global_accumulator_list = []
    self._combiner_init_op = None
    self._global_step = None
    self._sync_token_queue = None

    self._opt._prepare()  # pylint:disable=protected-access

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=1,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    return self._opt.compute_gradients(loss, var_list, gate_gradients, aggregation_method, \
                    colocate_gradients_with_ops, grad_loss)

  def _combiner_vars_update(self, var_list):
    """Get the update ops for the combiner variables in "var_list".

    Args:
      var_list: Optional list or tuple of 'tf.Variable' to update

    Returns:
      An update op

    Raises:
      ValueError: if var_list is empty.
    """
    if not var_list:
      raise ValueError("The list of combiner_variables should not be empty")
    update_ops = []
    global_center_vars = [self._combiner_2_global[var] for var in var_list]
    for combiner_var, global_var in zip(var_list, global_center_vars):
      with ops.device(combiner_var.device):
        update_ops.append(state_ops.assign(combiner_var, global_var.read_value()))
    return control_flow_ops.group(*(update_ops))

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer. The chief work updates global
    variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      A conditional 'Operation' that update both local and global variables or
      just local variables

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """

    # update local variables
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    train_ops = []
    aggregated_grad = []
    var_list = []

    # local_anchor op will be placed on this worker task by default.
    local_anchor = control_flow_ops.no_op()
    # Colocating local_step variable prevents it being placed on the PS.
    with ops.colocate_with(local_anchor):
      self._local_step = variable_scope.variable(
          initial_value=0,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          dtype=global_step.dtype.base_dtype,
          name="sync_rep_with_combiner_local_step")

    self.local_step_init_op = state_ops.assign(self._local_step, global_step)
    chief_init_ops = [self.local_step_init_op]
    controller_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
        variables.global_variables())

    for grad, var in grads_and_vars:
      # print ("var = ", var, "device = ", var.device)
      var_list.append(var)
      with ops.device(var.device):
        # Dense gradients.
        if grad is None:
          aggregated_grad.append(None)  # pass-through.
          continue
        elif isinstance(grad, ops.Tensor):
          # 每个var创建一个accumulator  
          grad_accum = data_flow_ops.ConditionalAccumulator(
              grad.dtype,
              shape=var.get_shape(),
              shared_name=var.name + "/grad_accum")
          # 梯度与local_step传入accumulator
          train_ops.append(grad_accum.apply_grad(
              grad, local_step=self._local_step))
          # 从accumulator中取出num_worker份梯度的均值，只要accumulator有，就会被取出
          aggregated_grad.append(grad_accum.take_grad(
              self._num_worker))
        
        #这部分和上面的大体一致，区别在于是整块插入。
        else:
          if not isinstance(grad, ops.IndexedSlices):
            raise ValueError("Unknown grad type!")
          grad_accum = data_flow_ops.SparseConditionalAccumulator(
              grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
          train_ops.append(grad_accum.apply_indexed_slices_grad(
              grad, local_step=self._local_step))
          aggregated_grad.append(grad_accum.take_indexed_slices_grad(
              self._num_worker))

        self._accumulator_list.append((grad_accum, var.device))

    # 取出的梯度均值与变量重新组成grads_and_vars
    aggregated_grads_and_vars = zip(aggregated_grad, var_list)

    # sync_op will be assigned to the same device as the global step.
    with ops.device(global_step.device), ops.name_scope(""):
      # base_opt用梯度均值实际更新变量
      update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                            global_step)

    def _update_global_variables():
      combiner_vars = [v for g, v in aggregated_grads_and_vars if g is not None]
      print("=====")
      print(combiner_vars)
      print(self._combiner_2_global)
      print("=====")
      global_vars = [self._combiner_2_global[v] for v in combiner_vars]

      # 创建同步队列sync queue
      # sync queue
      with ops.colocate_with(global_step):
        combiner_sync_queue = data_flow_ops.FIFOQueue(
            -1, [dtypes.bool], shapes=[[]], shared_name="global/sync_queue")
      combiner_train_ops = []
      combiner_aggregated_vars = []
      with ops.name_scope(None, self._name + "/global"):
        for var, gvar in zip(combiner_vars, global_vars):
          # pylint: disable=protected-access
          with ops.device(gvar.device):
            if isinstance(var._ref(), ops.Tensor):
              # 参考SyncReplicaOptimizer，将每个worker的local_var放入accumulator  
              combiner_var_accum = data_flow_ops.ConditionalAccumulator(
                  var.dtype,
                  shape=var.get_shape(),
                  shared_name="%s/%s/var_accum" % (self._combiner_field, gvar.name))
              combiner_train_ops.append(
                  combiner_var_accum.apply_grad(var._ref(), local_step=global_step))
              # 取出num_worker份local_var的平均值
              combiner_aggregated_vars.append(combiner_var_accum.take_grad(self._num_worker))
            else:
              raise ValueError("Unknown local variable type!")
            self._global_accumulator_list.append((combiner_var_accum, gvar.device))
      # chief worker负责将local_var的平均值赋值给global_var，然后在同步队列中加入num_combiner份令牌
      # chief worker updates global vars and enqueues tokens to the sync queue
      if self._is_chief:
        update_ops = []
        with ops.control_dependencies(train_ops):
          for avg_var, gvar in zip(combiner_aggregated_vars, global_vars):
            with ops.device(gvar.device):
              update_ops.append(state_ops.assign(gvar, avg_var))
          with ops.device(global_step.device):
            update_ops.append(state_ops.assign_add(global_step, 1))
        with ops.control_dependencies(update_ops), ops.device(
            global_step.device):
          tokens = array_ops.fill([self._num_combiner],
                                  constant_op.constant(False))
          sync_op = combiner_sync_queue.enqueue_many(tokens)
      # 其它controller的sync_op是从全局同步队列中取出一份token，当chief_worker没有完成global_var更新时，同步队列为空，因而其余controller只能等待，从而完成同步
      elif self._is_controller:
        with ops.control_dependencies(train_ops), ops.device(
            global_step.device):
          sync_op = combiner_sync_queue.dequeue()
      # 否则什么也不做
      else:
        sync_op = control_flow_ops.no_op()
      # 在完成global_var更新之后，所有combiner将global_var赋值给combiner_var
      with ops.control_dependencies([sync_op]):
        local_update_op = self._combiner_vars_update(combiner_vars)
      return local_update_op
    
    # Create token queue.
    with ops.device(self._combiner_device), ops.name_scope(""):
    
      # 同步队列，放入global_step
      sync_token_queue = (
          data_flow_ops.FIFOQueue(-1,
                                  global_step.dtype.base_dtype,
                                  shapes=(),
                                  name="sync_token_q",
                                  shared_name="%s/sync_token_q" % self._combiner_field))
      self._sync_token_queue = sync_token_queue

      # dummy_queue is passed to the queue runner. Don't use the real queues
      # because the queue runner doesn't automatically reopen it once it
      # closed queues in PS devices.
      dummy_queue = (
          data_flow_ops.FIFOQueue(1,
                                  types_pb2.DT_INT32,
                                  shapes=(),
                                  name="dummy_queue",
                                  shared_name="%s/dummy_queue" % self._combiner_field))

    with ops.device(self._combiner_device), ops.name_scope(""):

      # Replicas have to wait until they can get a token from the token queue.
      with ops.control_dependencies(train_ops):
        token = sync_token_queue.dequeue()
      train_op = state_ops.assign(self._local_step, token)

      # 与update_op强依赖，利用队列强制base_opt做更新 
      with ops.control_dependencies([update_op]):
        # Sync_op needs to insert tokens to the token queue at the end of the
        # step so the replicas can fetch them to start the next step.
        global_update_op = _update_global_variables()
        with ops.control_dependencies([global_update_op]):
          tokens = array_ops.fill([self._num_combiner], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))

      self._global_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                          [sync_op])
    for accum, dev in self._global_accumulator_list:
      with ops.device(dev):
        chief_init_ops.append(
            accum.set_global_step(
                global_step, name="SetGlobalStep"))
    for accum, dev in self._accumulator_list:
      with ops.device(dev):
        controller_init_ops.append(
            accum.set_global_step(
                global_step, name="SetGlobalStep"))
    self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
    self.controller_init_op = control_flow_ops.group(*(controller_init_ops))
    self._gradients_applied = True
    print("end function apply_gradients")
    return train_op

  def get_init_op(self):
    """Returns the op.

    This method lets all the local variables equal to the global
    variables before the training begins.
    """
    return self._combiner_vars_update(variables.trainable_variables())

  def get_global_queue_runner(self):
    """Returns the QueueRunner for the chief to execute.

    This includes the operations to synchronize replicas: aggregate gradients,
    apply to variables, increment global step, insert tokens to token queue.

    Note that this can only be called after calling apply_gradients() which
    actually generates this queuerunner.

    Returns:
      A `QueueRunner` for chief to execute.

    Raises:
      ValueError: If this is called before apply_gradients().
    """
    if self._gradients_applied is False:
      raise ValueError("Should be called after apply_gradients().")

    return self._global_queue_runner

  def make_session_run_hook(self):
    """Creates a hook to handle combiner ops such as initialization."""
    return _SyncReplicasWithCombinerOptimizerHook(self, self._is_chief, self._is_controller)

class _SyncReplicasWithCombinerOptimizerHook(session_run_hook.SessionRunHook):
  """A SessionRunHook handles ops related to SyncReplicasWithCombinerOptimizer."""

  def __init__(self, combiner_optimizer, is_chief, is_controller):
    """Creates hook to handle SyncReplicasWithCombinerOptimizer initialization ops.

    Args:
      sync_optimizer: `SyncReplicasWithCombinerOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
      is_controller: `Bool`, whether is this a combiner controller or not.
      num_tokens: Number of tokens to add to the queue.
    """
    self._combiner_optimizer = combiner_optimizer
    self._is_chief = is_chief
    self._is_controller = is_controller

  def begin(self):
    if self._combiner_optimizer._gradients_applied is False:  # pylint: disable=protected-access
      raise ValueError(
          "SyncReplicasOptimizer.apply_gradient should be called before using "
          "the hook.")
    # 如果是controller节点
    if self._is_controller:
      self._local_init_op = self._combiner_optimizer.controller_init_op
      self._q_runner = self._combiner_optimizer.get_global_queue_runner()
    # 如果这是一个chief节点
    elif self._is_chief:
      self._local_init_op = self._combiner_optimizer.chief_init_op
      self._q_runner = None
    else:
      self._local_init_op = self._combiner_optimizer.local_step_init_op
      self._q_runner = None

    self._ready_for_local_init_op = (
        self._combiner_optimizer.ready_for_local_init_op)

  def after_create_session(self, session, coord):
    """Runs SyncReplicasWithCombinerOptimizer initialization ops."""
    local_init_success, msg = session_manager._ready(  # pylint: disable=protected-access
        self._ready_for_local_init_op, session,
        "Model is not ready for SyncReplicasOptimizer local init.")
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for SyncReplicasOptimizer "
          "local_init. Init op: %s, error: %s" %
          (self._local_init_op.name, msg))
    # 执行初始化
    session.run(self._local_init_op)
    # 如果是combiner，执行队列
    if self._q_runner is not None:
      self._q_runner.create_threads(
          session, coord=coord, daemon=True, start=True)