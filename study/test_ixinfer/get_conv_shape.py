import onnx
from onnx import shape_inference

def get_attribute_dict(item_attribute):
    attribute_dict = {}
    for attr in item_attribute:
        if attr.type == onnx.AttributeProto().AttributeType.FLOAT:
            attribute_dict[attr.name] = attr.f
        if attr.type == onnx.AttributeProto().AttributeType.FLOATS:
            attribute_dict[attr.name] = [x for x in attr.floats]
        if attr.type == onnx.AttributeProto().AttributeType.INT:
            attribute_dict[attr.name] = attr.i
        if attr.type == onnx.AttributeProto().AttributeType.INTS:
            attribute_dict[attr.name] = [x for x in attr.ints]
        if attr.type == onnx.AttributeProto().AttributeType.STRING:
            attribute_dict[attr.name] = str(attr.s.decode('UTF-8'))
        if attr.type == onnx.AttributeProto().AttributeType.STRINGS: 
            attribute_dict[attr.name] = [str(x.decode('UTF-8')) for x in attr.strings]
    return attribute_dict

def get_yolov5_conv_shape(onnx_path):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    input_tensor = graph.input[0]
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
    inferred_graph = inferred_onnx_model.graph
    initializer_tensors = inferred_onnx_model.graph.initializer
    tensor_shape_dict = {}
    for init_tensor in initializer_tensors:
        tensor_shape_dict[init_tensor.name] = init_tensor.dims
    nodes = inferred_graph.node
    feature_map_size_dict = {}
    feature_map_size_dict['images'] = 640
    conv_attr_list = []
    for i, item in enumerate(nodes):
        input_name = item.input[0]
        output_name = item.output[0]
        feature_map_size_dict[output_name] = feature_map_size_dict[input_name]
        if item.op_type == 'Conv':
            weights_name = item.input[1]
            weights_dims = tensor_shape_dict[weights_name]
            output_channel = weights_dims[0]
            input_channel = weights_dims[1]
            if input_channel == 3:
                input_channel = 4
            kernel_h = weights_dims[2]
            kernel_w = weights_dims[3]

            attr_dict = get_attribute_dict(item.attribute)
            conv_stride = attr_dict['strides'][0]
            conv_padding = attr_dict['pads'][0]
            if conv_stride == 2:
                feature_map_size_dict[output_name] = feature_map_size_dict[input_name]//2
            input_size = feature_map_size_dict[input_name]
            output_size = feature_map_size_dict[output_name]
            conv_attr=[32,input_size,input_size,input_channel,kernel_h,kernel_w,output_channel,output_size,output_size,conv_padding,conv_stride]
            if not (conv_attr in conv_attr_list):
                print(item.name,conv_attr)
                conv_attr_list.append(conv_attr)
        if item.op_type == 'Resize':
            feature_map_size_dict[output_name] = feature_map_size_dict[input_name]*2

# int compare_qdconv(int batch_size, int input_h, int input_w, int input_c, int kernel_h, int kernel_w, int  output_c, int output_h, int output_w, int input_pad, int input_stride,bool use_pechannel, bool use_bias,float leaky_gamma)

if __name__ == '__main__':
    get_yolov5_conv_shape('/home/feihu.fang/inference/libinfer/model_convert/yolov5/quantized_yolov5m.onnx')