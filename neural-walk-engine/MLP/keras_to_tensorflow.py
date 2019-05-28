from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


import argparse

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-weights', type=str, default='weights')
ap.add_argument('-output', type=str, default='graph_model')
args = vars(ap.parse_args())


input_fld = os.getcwd() 
weight_file = args['weights']
num_output = 1
write_graph_def_ascii_flag = False
prefix_output_node_names_of_final_network = 'pi/output_node'
output_fld = input_fld
output_graph_name = args['output'] + '.pb'


if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)
net_model = load_model(weight_file_path)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = 	prefix_output_node_names_of_final_network
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()

if write_graph_def_ascii_flag:
    f = 'only_the_graph_def.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))
