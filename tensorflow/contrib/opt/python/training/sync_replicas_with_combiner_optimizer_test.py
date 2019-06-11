# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ModelAverageOptimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import portpicker

from tensorflow.contrib.opt.python.training import sync_replicas_with_combiner_optimizer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import server_lib
from tensorflow.python.training import training
from tensorflow.python.training import training_util


def create_local_cluster(num_workers, num_ps, protocol="grpc"):
  """Create local GRPC servers and return them."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = server_lib.ClusterSpec(cluster_dict)

  workers = [
      server_lib.Server(
          cs, job_name="worker", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs, job_name="ps", protocol=protocol, task_index=ix, start=True)
      for ix in range(num_ps)
  ]

  return cluster_dict, workers, ps_servers

class ModelAverageOptimizerTest(test.TestCase):

  def _run(self, train_op, sess):
    sess.run(train_op)

  def testPS2TasksWithClusterSpecClass(self):
    cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "combiner": ["combiner0:2222", "combiner1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"],
    })
    worker_device = "/job:worker/task:0"
    combiner_device = "/job:combiner/task:1"
    combiner_field = "combiner0"
    psc_coustom = sync_replicas_with_combiner_optimizer.CombinerCustomGetter(combiner_device=combiner_device, combiner_field=combiner_field)
    from tensorflow.python.training import device_setter
    with ops.device(
        device_setter.replica_device_setter(cluster=cluster_spec,
                                            worker_device=worker_device,
                                            ps_device="/job:ps")), \
         variable_scope.variable_scope("", custom_getter=psc_coustom):
      v = variable_scope.get_variable(initializer=[1, 2], name="v")
      w = variable_scope.get_variable(initializer=[2, 1], name="w")
      global_step = training_util.get_or_create_global_step()
      v_g, w_g = psc_coustom._combiner_2_global[v], psc_coustom._combiner_2_global[w]
      self.assertDeviceEqual("/job:combiner/task:1", v.device)
      self.assertDeviceEqual("job:ps/task:0", v_g.device)
      self.assertDeviceEqual("/job:combiner/task:1", w.device)
      self.assertDeviceEqual("job:ps/task:1", w_g.device)


if __name__ == "__main__":
  test.main()
