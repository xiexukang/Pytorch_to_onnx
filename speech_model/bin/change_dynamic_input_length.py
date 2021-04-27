import onnx
import os
import onnx.checker
import onnx.utils
from onnx.tools import update_model_dims
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--onnx_model_path',
                        required=True,
                        help='encoder model')

    args = parser.parse_args()

    encoder_path = os.path.join(args.onnx_model_path, 'fixed_dim_encoder.onnx')

    decoder_init_path = os.path.join(args.onnx_model_path,
                                     'fixed_dim_decoder_init.onnx')

    decoder_non_init_path = os.path.join(args.onnx_model_path,
                                         'fixed_dim_decoder_non_init.onnx')

    dynamic_encoder_path = os.path.join(args.onnx_model_path,
                                        'dynamic_dim_encoder.onnx')

    dynamic_decoder_init_path = os.path.join(args.onnx_model_path,
                                             'dynamic_dim_decoder_init.onnx')

    dynamic_decoder_non_init_path = os.path.join(
        args.onnx_model_path, 'dynamic_dim_decoder_non_init.onnx')

    model = onnx.load(encoder_path)
    dim_proto0 = model.graph.input[0].type.tensor_type.shape.dim[1]
    dim_proto0.dim_param = 'input.0_1'
    dim_proto_o1 = model.graph.output[0].type.tensor_type.shape.dim[1]
    dim_proto_o1.dim_param = 'output.0_1'
    dim_proto_o2 = model.graph.output[1].type.tensor_type.shape.dim[2]
    dim_proto_o2.dim_param = 'output.1_2'
    onnx.save(model, dynamic_encoder_path)

    model = onnx.load(decoder_init_path)
    dim_proto0 = model.graph.input[0].type.tensor_type.shape.dim[1]
    dim_proto0.dim_param = 'input.0_1'
    dim_proto0 = model.graph.input[1].type.tensor_type.shape.dim[2]
    dim_proto0.dim_param = 'input.1_2'
    onnx.save(model, dynamic_decoder_init_path)

    model = onnx.load(decoder_non_init_path)
    #input[0]: T1 * 512  <-- 17 * 512
    #input[1]: T2  <-- (2)
    #input[2] -- input[7]: 1, T3, 512 <-- 1, 1, 512
    # 0
    dim_proto0 = model.graph.input[0].type.tensor_type.shape.dim[1]
    dim_proto0.dim_param = 'input.0_1'

    dim_proto1 = model.graph.input[1].type.tensor_type.shape.dim[2]
    dim_proto1.dim_param = 'input.1_2'

    dim_proto2 = model.graph.input[2].type.tensor_type.shape.dim[1]
    dim_proto2.dim_param = 'input.2_1'

    dim_proto3 = model.graph.input[3].type.tensor_type.shape.dim[1]
    dim_proto3.dim_param = 'input.3_1'
    dim_proto3 = model.graph.input[3].type.tensor_type.shape.dim[2]
    dim_proto3.dim_param = 'input.3_2'

    dim_proto4 = model.graph.input[4].type.tensor_type.shape.dim[1]
    dim_proto4.dim_param = 'input.4_1'

    dim_proto5 = model.graph.input[5].type.tensor_type.shape.dim[1]
    dim_proto5.dim_param = 'input.5_1'

    dim_proto6 = model.graph.input[6].type.tensor_type.shape.dim[1]
    dim_proto6.dim_param = 'input.6_1'

    dim_proto7 = model.graph.input[7].type.tensor_type.shape.dim[1]
    dim_proto7.dim_param = 'input.7_1'

    dim_proto7 = model.graph.input[8].type.tensor_type.shape.dim[1]
    dim_proto7.dim_param = 'input.8_1'

    dim_proto7 = model.graph.input[9].type.tensor_type.shape.dim[1]
    dim_proto7.dim_param = 'input.9_1'

    dim_proto_o1 = model.graph.output[1].type.tensor_type.shape.dim[1]
    dim_proto_o1.dim_param = 'output.1_1'

    dim_proto_o1 = model.graph.output[2].type.tensor_type.shape.dim[1]
    dim_proto_o1.dim_param = 'output.2_1'

    dim_proto_o1 = model.graph.output[3].type.tensor_type.shape.dim[1]
    dim_proto_o1.dim_param = 'output.3_1'

    dim_proto_o1 = model.graph.output[4].type.tensor_type.shape.dim[1]
    dim_proto_o1.dim_param = 'output.4_1'

    dim_proto_o1 = model.graph.output[5].type.tensor_type.shape.dim[1]
    dim_proto_o1.dim_param = 'output.5_1'

    dim_proto_o1 = model.graph.output[6].type.tensor_type.shape.dim[1]
    dim_proto_o1.dim_param = 'output.6_1'
    onnx.save(model, dynamic_decoder_non_init_path)
