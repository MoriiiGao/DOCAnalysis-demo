# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
https://blog.csdn.net/weixin_43694096/article/details/124695537
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda import amp

from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode, time_sync


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, dilation=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # p = (int((k - 1) / 2) * dilation, int((k - 1) / 2) * dilation)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False,dilation=dilation)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution class
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # for i in x:
        #     print(i.shape)
        return torch.cat(x, self.d)

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self._model_type(w)  # get backend
        w = attempt_download(w)  # download if not local
        fp16 &= pt or jit or onnx or engine  # FP16
        stride = 32  # default stride

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available() and device.type != 'cpu'
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
            output_layer = next(iter(executable_network.outputs))
            meta = Path(w).with_suffix('.yaml')
            if meta.exists():
                stride, names = self._load_metadata(meta)  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            fp16 = False  # default updated below
            dynamic = False
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                if model.binding_is_input(index):
                    if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        fp16 = True
                shape = tuple(context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # graph_def
                with open(w, 'rb') as f:
                    gd.ParseFromString(f.read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {
                        'Linux': 'libedgetpu.so.1',
                        'Darwin': 'libedgetpu.1.dylib',
                        'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
            elif tfjs:
                raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')
            else:
                raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else [f'class{i}' for i in range(999)]
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
            if isinstance(y, tuple):
                y = y[0]
        elif self.jit:  # TorchScript
            y = self.model(im)[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = self.executable_network([im])[self.output_layer]
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
                y = y[k]  # output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = (self.model(im, training=False) if self.keras else self.model(im)).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            else:  # Lite or Edge TPU
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb
        if any(warmup_types) and self.device.type != 'cpu':
            im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs

    @staticmethod
    def _load_metadata(f='path/to/meta.yaml'):
        # Load metadata from meta.yaml if it exists
        d = yaml_load(f)
        return d['stride'], d['names']  # assign stride, names


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1, device=self.model.device)  # for device, type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), list(imgs)) if isinstance(imgs, (list, tuple)) else (1, [imgs])  # number, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) if self.pt else size for x in np.array(shape1).max(0)]  # inf shape
        x = [letterbox(im, shape1, auto=False)[0] for im in imgs]  # pad
        x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0],
                                    self.conf,
                                    self.iou,
                                    self.classes,
                                    self.agnostic,
                                    self.multi_label,
                                    max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self, labels=True):
        self.display(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self.display(render=True, labels=labels)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n  # override len(results)

    def __str__(self):
        self.print()  # override print(results)
        return ''


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True,dilation=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False,dilation=dilation)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置偏差的参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # 获取窗口内每个标记的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw], indexing="ij"
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerLayer(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if num_heads > 10:
            drop_path = 0.1
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(c)
        self.attn = WindowAttention(
            c, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c)
        mlp_hidden_dim = int(c * mlp_ratio)
        self.mlp = Mlp(in_features=c, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = ((0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, torch.tensor(-100.0)).masked_fill(attn_mask == 0,
                                                                                            torch.tensor(0.0))
        return attn_mask

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()  # [b,h,w,c]

        attn_mask = self.create_mask(x, h, w)  # [nW, Mh*Mw, Mh*Mw]
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            # print(f"shift size: {self.shift_size}")
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [B, H', W', C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 3, 2, 1).contiguous()
        return x  # (b, self.c2, w, h)

class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2#4
        self.tr = nn.Sequential(*(SwinTransformerLayer(c2, num_heads=num_heads, window_size=window_size,  shift_size=0 if (i % 2 == 0) else self.shift_size ) for i in range(num_layers)))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.tr(x)
        return x


class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)#c1, c2, num_heads, num_layers

class SeparableConv(nn.Module):
    '''c1, c2, k=1, s=1, p=None, g=1, act=True,dilation=1'''
    def __init__(self, c1, c2, k, s=1, dilation=1, act=True):
        super(SeparableConv, self).__init__()

        self.dw_conv = Conv(c1, c1, k=k, s=s, dilation=dilation, g=c1, act=False)
        self.pw_conv = Conv(c1, c2, k=1, s=1, dilation=1, g=1, act=True)

    def forward(self,x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

class LFEM(nn.Module):
    def __init__(self,c1,c2,output_size,stride=2):
        super(LFEM,self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

        self.coordConv = CoordConv(c1,c2)

        self.low_proj = Conv(c1,c2)
        # self.AvgPool = nn.AdaptiveAvgPool2d(output_size)
        # self.MaxPool = nn.AdaptiveMaxPool2d(output_size)
        self.AvgPool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.MaxPool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.Pool_proj = Conv(c2,c2)

    def forward(self, x):

        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        fm = out * x
        return fm

        # lf = self.low_proj(x)#[1,96,64,64]
        #
        # lf1 = self.AvgPool(lf)
        # lf1 = self.Pool_proj(lf1).transpose(1, 3)
        #
        # lf2 = self.MaxPool(lf)
        # lf2 = self.Pool_proj(lf2).transpose(1, 3)
        #
        # lf3 = F.softmax(torch.mul(lf1, lf2), dim=-1)#[1,32,32,96]
        #
        # # lf = torch.mul(lf3,lf.transpose(1,3))
        #
        # lf1 = torch.mul(lf1, lf3)
        # lf2 = torch.mul(lf2, lf3)
        #
        # lf = torch.add(lf1, lf2).transpose(1, 3)
        # return lf

class DilaFormer(nn.Module):
    def __init__(self, c1, c2, dilation, k=3):
        super(DilaFormer, self).__init__()

        padding = (int((k - 1) / 2) * dilation, int((k - 1) / 2) * dilation)
        self.dilaConv = SeparableConv(c1, c2, k=k)
        # self.dilaConv = Conv(c1, c2, k=k, p=padding, dilation=dilation)
        # self.conv = Conv(c2,c2,k=1)
        # self.mlp = MLP(c2,c2)

    def forward(self,x):

        # B,C,H,W = x.shape

        x = self.dilaConv(x)
        # x = x + self.conv(self.dilaConv(x))
        # x = x + self.mlp(x.reshape(B,H*W,C),H,W).reshape(B,C,H,W)

        return x

class SFEM(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super(SFEM, self).__init__()

        # self.dila_conv1 = DilaFormer(c1, c2, dilation=1)
        # self.dila_conv2 = DilaFormer(c1, c2, dilation=2)
        # self.dila_conv3 = DilaFormer(c1, c2, dilation=4)
        self.dila_conv1 = DilaFormer(c1, c2, k=3, dilation=1)
        self.dila_conv2 = DilaFormer(c1, c2, k=5, dilation=1)
        self.dila_conv3 = DilaFormer(c1, c2, k=7, dilation=1)
        self.branch_proj = Conv(c2, c2)
        self.lfem = LFEM(c2, c2, 40)

    def forward(self,x):
        B,C,W,H = x.shape[:]
        # print(W,H)
        #x:[BZ,32,128,128]
        branch1 = self.dila_conv1(x)
        branch2 = self.dila_conv2(x)
        branch3 = self.dila_conv3(x)

        # print("b1:", branch1.shape)
        # print("b2:", branch2.shape)
        # print("b3:", branch3.shape)
        branch1 = self.lfem(branch1)
        branch2 = self.lfem(branch2)
        branch3 = self.lfem(branch3)

        fm = self.branch_proj(torch.add(torch.add(branch1, branch2), branch3))

        return fm

class SFEM1(nn.Module):
    def __init__(self, c1, c2):
        super(SFEM1,self).__init__()
        self.cv1 = Conv(c1, c2, k=3, p=1)
        self.cv2 = Conv(c1, c2, k=3, dilation=3, p=3)
        self.cv3 = Conv(c1, c2, k=3, dilation=5, p=5)
        self.cv4 = Conv(c1, c2, k=3, dilation=7, p=7)

    def forward(self,x):
        res = x
        # print(res.shape)
        print(self.cv1)
        print(self.cv2)
        print(self.cv3)
        print(self.cv4)
        return res + self.cv1(x) + self.cv2(x) + self.cv3(x) + self.cv4(x)

class SFEM2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(SFEM2, self).__init__()

        c_ = int(c2 * e)
        self.dila_conv1 = nn.Conv2d(c1, c_, kernel_size=3, dilation=1, padding=1, groups=c_)
        self.dila_conv2 = nn.Conv2d(c1, c_, kernel_size=3, dilation=3, padding=3, groups=c_)
        self.dila_conv3 = nn.Conv2d(c1, c_, kernel_size=3, dilation=5, padding=5, groups=c_)
        # self.dila_conv4 = nn.Conv2d(c1, c_, kernel_size=3, dilation=7, padding=7)
        self.fusion_proj = Conv(c1, c_, 1)
        self.CAM = ChannelAttentionModule(c_, c_)
        self.SAM = SpatialAttentionModule(c_, c_)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.cv1 = Conv(2 * c_, c2, 1)
        # self.fusion_proj1 = SpatialAttentionModule(c_ * 2, c2)


    def forward(self, x):

        # branch1 = self.dila_conv1(x)
        # branch2 = self.dila_conv2(x)
        # branch3 = self.dila_conv3(x)
        # # branch4 = self.dila_conv4(x)
        #
        # branch = torch.cat([branch1, branch2, branch3], 1)
        branch = self.fusion_proj(x)

        branch1 = self.m(self.SAM(branch))
        branch2 = self.SAM(branch)

        branch = self.cv1(torch.cat([branch1, branch2], dim=1))

        return branch

class MLP(nn.Module):
    def __init__(self,c1,c2,drop=0.):
        super(MLP, self).__init__()

        self.bn = nn.BatchNorm1d(c1)
        self.fc1 = nn.Conv2d(c1,c2,1,bias=True)
        self.act = nn.ReLU()
        # self.conv = DWConv(c2,c2,act=True)
        self.fc2 = nn.Conv2d(c2,c1,1,bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self,x,H,W):

        B,N,C = x.shape[:]
        # W = H = int(pow(N,0.5))

        x = self.fc1(self.bn(x.transpose(-1,-2)).reshape(B,C,H,W))
        x = self.fc2(self.act(x)).reshape(B,N,C)
        # x = self.conv(x.reshape(B,C,H,W)).reshape(B,H*W,C)
        # x = self.conv(x)
        # x = self.dwconv(x.reshape(B,C,H,W)).reshape(B,C,-1).transpose(1,2)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = sel
        # f.drop(x)

        return x

class SKAttention(nn.Module):
    def __init__(self, c1, c2, kernels=[1, 3, 5, 7], reductions=16, group=1, L=32):
        super(SKAttention, self).__init__()
        if c1 == c2:
            self.d = max(L, c1//reductions)
        else:
            self.d = L

        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ("Conv", nn.Conv2d(c1, c1, kernel_size=k, padding=k//2, groups=group)),
                    ("BN", nn.BatchNorm2d(c1)),
                    ("SiLU", nn.SiLU())
                ]))
            )

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fusion = Conv(c1, c2)

        self.fc = Conv(c1, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(Conv(self.d, c1))
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape[:]
        conv_outs = list()

        ###split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)#[K, B, C, H, W]

        ### fuse
        U = sum(conv_outs)#[B, C, H, W]
        max_U = self.maxpool(U)
        avg_U  = self.avgpool(U)
        S = self.fusion(torch.cat([max_U, avg_U], dim=1))

        #reduction channel
        Z = self.fc(S)#B, d

        #calculate attention weight
        weights = list()
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight)
        attention_weights = torch.stack(weights, 0)#[k, B, C, 1, 1]
        attention_weights = self.sigmoid(attention_weights)

        ###fuse
        V = (attention_weights * feats).sum(0)

        return V

def INF(B, W, H):
    return torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B*W, 1, 1)


class Attention(nn.Module):
    def __init__(self, c1, c2, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, channel_ratio=4):
        super(Attention, self).__init__()
        assert c2 % num_heads == 0, f"dim {c2} should be devided by {num_heads}"

        self.channel_ratio = channel_ratio
        self.dim = c2
        self.num_heads = num_heads
        head_dim = c2 // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = Conv(c1, c2)
        self.kv = Conv(c1, (c2//self.channel_ratio)*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_features=c2, out_features=c2)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

    def forward(self, x, y, H, W):
        B, N, C = x.shape

        x, y = x.reshape(B, C, H, W), y.reshape(B, C, H, W)
        #q:[B,h,N,c]
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)

        #k,v:[B,h,N,c]
        k, v = kv[0], kv[1]

        #1.注意力得分:放缩点积计算qk相似度
        attn = (q * self.scale) @ k.transpose(-2, -1)

        #2.softmax对注意力得分进行数值转换：归一化得到所有全重系数之和为1的概率分布，突出重要像素的全重
        att_map = attn.softmax(dim=-1)
        att_map = self.attn_drop(att_map)

        x = (att_map @ v).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SSAttention(nn.Module):
    """Spatial-Select-Attention"""
    def __init__(self, c1, c2, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super(SSAttention, self).__init__()
        assert c2 % num_heads == 0, f"dim {c2} should be devided by {num_heads}"

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.SpatialConv = Conv(c1, c1)
        self.sig = nn.Sigmoid()

        self.fusion = Conv(c1*2, c2)

    def forward(self, x, y, H, W):
        B, N, C = x.shape

        x, y = x.reshape(B, C, H, W), y.reshape(B, C, H, W)

        fm = torch.add(x, y)

        att_map = self.sig(self.SpatialConv(self.maxpool(fm).view(B, C, 1, 1)))

        #feat_x
        feat_x = x*att_map.expand_as(x)
        #faet_y
        feat_y = y*att_map.expand_as(y)

        fm = self.fusion(torch.cat([feat_x, feat_y], dim=1))

        return fm.view(B, N, C)

class SEAttention(nn.Module):
    """Sequeeze-and-Extract Attention"""
    def __init__(self, c1, c2, ratio=16, act=True, bias=False):
        super(SEAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=bias)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.l2 = nn.Linear(c1 // ratio, c1, bias=bias)
        self.sig = nn.Sigmoid()

        self.fusion = Conv(c1, c2)

    def forward(self, x, y, H, W):
        B, N, C = x.shape
        x, y = x.reshape(B, C, H, W), y.reshape(B, C, H, W)

        res_x = self.avgpool(x).view(B, C)
        res_x = self.sig(self.l2(self.act(self.l1(res_x))))
        res_x = res_x.view(B, C, 1, 1)
        att_x = x * res_x.expand_as(x)

        res_y = self.maxpool(y).view(B, C)
        res_y = self.sig(self.l2(self.act(self.l1(res_y))))
        res_y = res_y.view(B, C, 1, 1)
        att_y = y * res_y.expand_as(y)

        fusion_map = self.fusion(torch.add(att_x, att_y)).view(B, N, C)

        return fusion_map


class Multi_Query_Attention(nn.Module):
    def __init__(self, c1, c2, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, channel_ratio=4):
        super(Multi_Query_Attention, self).__init__()
        assert c2 % num_heads == 0, f"dim {c2} should be devided by {num_heads}"

        self.channel_ratio = channel_ratio
        self.dim = c2
        self.num_heads = num_heads
        head_dim = c2 // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = Conv(c1, c2)
        self.kv = Conv(c1, (c2 // self.channel_ratio) * 2)
        # self.kv = Conv(c1, c2*2)
        # self.q = nn.Linear(in_features=c1, out_features=c2, bias=qkv_bias)
        # self.kv = nn.Linear(in_features=c1, out_features=c2 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_features=c2, out_features=c2)
        self.proj = Conv(c2, c2)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

    def forward(self, x, y, H, W):
        B, N, C = x.shape

        x, y = x.reshape(B, C, H, W), y.reshape(B, C, H, W)
        # q:[B,h,N,c]
        q = self.q(x).reshape(B, N, self.num_heads, ((C // self.channel_ratio) // self.num_heads), self.channel_ratio).permute(4, 0, 2, 1, 3)
        q_list = [q[i] for i in range(self.channel_ratio)]

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, (C // self.channel_ratio) // self.num_heads).permute(2, 0, 3, 1, 4)

        # k,v:[B,h,N,c]
        k, v = kv[0], kv[1]

        att_x = list()
        for q in q_list:
            # 1.注意力得分:放缩点积计算qk相似度
            attn = (q * self.scale) @ k.transpose(-2, -1)

            # 2.softmax对注意力得分进行数值转换：归一化得到所有全重系数之和为1的概率分布，突出重要像素的全重
            att_map = attn.softmax(dim=-1)
            att_map = self.attn_drop(att_map)

            x = (att_map @ v).reshape(B, N, C//self.channel_ratio)
            att_x.append(x)

        x = self.proj(torch.cat(att_x, dim=2).reshape(B, C, H, W))
        x = self.proj_drop(x.reshape(B, N, C))
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, c2, output_size=1, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        mid_channels = c1 // reduction

        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.MaxPool = nn.AdaptiveMaxPool2d(output_size=output_size)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, c2, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        res = x

        avgout = self.shared_MLP(self.AvgPool(x))

        maxout = self.shared_MLP(self.MaxPool(x))

        return self.sigmoid(torch.add(avgout, maxout)) * res

class SpatialAttentionModule(nn.Module):
    def __init__(self, c1, c2):

        super(SpatialAttentionModule, self).__init__()
        self.conv = Conv(c1=2, c2=1, k=7, s=1, p=3)
        self.sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm2d(c2)

    def forward(self,x):

        res = x

        #map尺寸不变 缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)

        maxout,_ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avgout, maxout], dim=1)

        out = self.sigmoid(self.conv(out)) * res

        out = self.BN(out)

        return out

class SAMBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(SAMBottleneck,self).__init__()

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.SAM = SpatialAttentionModule()

    def forward(self, x):
        return x + self.SAM(self.cv2(self.cv1(x))) if self.add else self.SAM(self.cv2(self.cv1(x)))


class CBAM(nn.Module):
    def __init__(self, c1, c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1,c1)
        self.spatial_attention = SpatialAttentionModule()
        self.relu = nn.ReLU()
        self.conv = Conv(c1, c2)
        self.layer_norm = nn.LayerNorm(c2)

    def forward(self,x):

        res = x
        out = self.channel_attention(x) * x#[1,368,32,32]
        out = self.spatial_attention(out) * out
        out = res + out

        out = self.relu(out)
        out = self.conv(out)
        out = self.layer_norm(out.transpose(1, 3)).transpose(1, 3)
        out = self.relu(out)
        return out

class Transformer_Block(nn.Module):
    def __init__(self, c1, c2, drop_path=0.3, sr_ratio=1):
        super(Transformer_Block, self).__init__()

        self.cbam = CBAM(c1, c1)
        self.norm1 = nn.LayerNorm(normalized_shape=c1)
        self.attn = Attention(c1, c2)
        self.drop_path = nn.Dropout(drop_path)

        self.norm2 = nn.LayerNorm(normalized_shape=c2)
        self.mlp = MLP(c2, c2)

    def forward(self,input):

        if len(input) == 3:
            z, x, y = input[:]
        else:
            x, y = input[:]

        B, C, H, W = x.shape[0:]

        x = torch.add(x,y)

        fm = self.cbam(x)

        x_ = x.reshape(B,C,-1).transpose(1,2)

        y_ = fm.reshape(B,C,-1).transpose(1,2)

        x = x_ + self.drop_path(
                 self.attn(
                 self.norm1(x_), self.norm1(y_), H, W))

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        x = x.transpose(1, 2)

        x = x.reshape(B, C, H, W)

        return x

class Transformer_Block1(nn.Module):
    def __init__(self, c1, c2, drop_path=0., sr_ratio=1):
        super(Transformer_Block1, self).__init__()
        self.bn = nn.BatchNorm2d(c1)
        # self.cbam = CBAM(c1, c1)
        self.norm1 = nn.LayerNorm(normalized_shape=c1)
        self.attn = SEAttention(c1, c2)
        # self.attn = Multi_Query_Attention(c1, c2)
        self.drop_path = nn.Dropout(drop_path)

        self.norm2 = nn.LayerNorm(normalized_shape=c2)
        self.mlp = MLP(c2, c2)

    def forward(self, x, y, H, W):


        # x, y = [self.norm1(i) for i in [x, y]][:]
        x = self.attn(x, y, H, W)
        x = self.drop_path(x)

        # x = x + self.drop_path(self.attn(self.norm1(x)))

        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # x = x.transpose(1, 2)

        # x = x.reshape(B,C,H,W)

        return x

class Patch_Transformer_Block(nn.Module):
    '''c1:128 c2:256'''
    def __init__(self, c1, c2, patch_h=4, patch_w=4):
        super(Patch_Transformer_Block, self).__init__()
        # self.bn = nn.BatchNorm2d(c2)
        print(c1, c2)
        conv_1x1_in = Conv(c1, c1//2)
        # conv_3x3_in = Conv(c1//2, c1//2, k=3)
        conv_1x1_out = Conv(c1//2, c2)

        # self.cbam = CBAM(c1, c1)

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv1x1", module=conv_1x1_in)
        # self.local_rep.add_module(name='conv3x3', module=conv_3x3_in)

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_h * patch_w

        # self.globel_rep = Transformer_Block1(c1//2, c1//2)
        self.globel_rep = SSAttention(c1//2, c2//2)
        self.conv_proj = conv_1x1_out

        self.fusion = Conv(c2, c2)

    def unfolding(self, feature_map):

        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape[:]

        new_h = int(math.ceil(orig_h // self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w // self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        #patch的宽高和个数
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, input):

        x, y = input[:]

        #reduce channels c1-->c1/2
        local_x = self.local_rep(x)
        local_y = self.local_rep(y)

        #token to patch
        patches_x, info_dict_x = self.unfolding(local_x)
        patches_y, info_dict_y = self.unfolding(local_y)

        H, W = info_dict_x["num_patches_h"], info_dict_x["num_patches_w"]

        patches = self.globel_rep(patches_x, patches_y, H, W)

        fm = self.folding(patches, info_dict_x)

        fm = self.conv_proj(fm)

        fm = self.fusion(x + y + fm)

        return fm

class SimConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,groups=1,bias=False):
        super(SimConv,self).__init__()
        padding = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self,x):
        return self.act(self.conv(x))

class SimSPPF(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5):
        super(SimSPPF,self).__init__()
        c_ = in_channels // 2
        self.cv1 = SimConv(in_channels, c_, kernel_size=1, stride=1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size,stride=1,padding=kernel_size // 2)

    def forward(self,x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, padding, stride=1,  dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False, padding=0),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False, padding=0),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False, padding=0),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False, padding=0)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class CoordSpatialAttention(nn.Module):
    def __init__(self):
        super(CoordSpatialAttention, self).__init__()
        self.conv = Conv(c1=2, c2=1, k=7, s=1, p=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        res = x

        x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device).to(torch.float16)
        y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device).to(torch.float16)
        # print(x.dtype,x_range.dtype)
        y_range, x_range = torch.meshgrid(y_range, x_range)

        y_range = y_range.expand([x.shape[0], 1, -1, -1])
        x_range = x_range.expand([x.shape[0], 1, -1, -1])
        # print(y_range.dtype, x_range.dtype)

        avgout = torch.mean(x, dim=1, keepdim=True)

        maxout, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avgout, maxout], dim=1)

        out = self.sigmoid(self.conv(out)) * res + res

        return out

class CBAMBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5,ratio=16,kernel_size=7):  # ch_in, ch_out, shortcut, groups, expansion
        super(CBAMBottleneck,self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        #self.cbam=CBAM(c1,c2,ratio,kernel_size)

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        # out = self.channel_attention(x1) * x1
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(x1) * x1
        return x + out if self.add else out

class C3CBAM(C3):
    # C3 module with CBAMBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, p=1):
        super(CoordConv,self).__init__()
        self.proj = Conv(in_channels+2, out_channels, k=kernel_size, dilation=dilation, p=p)

    def forward(self, x):
        ins_feat = x

        #生成从-1到1的线性值
        x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device).to(torch.float16)
        y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device).to(torch.float16)

        #生成二维坐标网络
        y, x = torch.meshgrid(y_range, x_range)

        #扩充到和ins_feat相同维度
        y = y.expand([ins_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_feat.shape[0], 1, -1, -1])

        #位置特征
        coord_feat = torch.cat([y, x], dim=1)

        #原始特征 位置特征合并
        ins_feat = torch.cat([ins_feat, coord_feat], 1)

        fm = self.proj(ins_feat)

        return fm

class CatConv(nn.Module):
    def __init__(self, c1, c2):
        super(CatConv, self).__init__()

        self.BConv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.BConv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        res = self.BConv1_1(x)
        res1 = self.BConv3_3(self.BConv3_3(res))


class ChannelAttention(nn.Module):
    def __init__(self, c1, reduction=4):
        super(ChannelAttention, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.sigmoid = nn.Sigmoid()
        #self.act=SiLU()
    def forward(self, x):

        res = x

        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)

        out = self.sigmoid(torch.add(avgout, maxout)) * res

        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv(torch.cat([avgout, maxout], dim=1)))

        return out

class SpatialFeatureRefine(nn.Module):
    def __init__(self, c1, c2, n=1, e=32, shortcut=True):
        super(SpatialFeatureRefine, self).__init__()
        '''
        c1, c2, n=1
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        '''
        # assert c1 == c2
        # hidden = c1 // 32
        # print('hidden:',hidden*4)

        self.ChannelAttention = ChannelAttention(c1)
        self.SpatialAttention = SpatialAttention()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.BConv1 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=1,padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.BConv2 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.BConv3 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=c2*4, out_channels=c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        # self.branch1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=1, padding=1)
        # self.branch2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=2, padding=2)
        # self.branch3 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=4, padding=4)
        # self.fusion = nn.Conv2d(in_channels=c2*3, out_channels=c2, kernel_size=1, stride=1)
        # self.act = nn.LeakyReLU()
        # self.BN = nn.BatchNorm2d(c2)
        # self.C3 = C3(c2, c2, n=n, shortcut=shortcut)

    def forward(self, x):

        x = self.ChannelAttention(x)

        att_map = self.SpatialAttention(x)

        x = x * att_map

        y1 = self.BConv1(x) * att_map

        y2 = self.BConv2(x) * att_map

        y3 = self.BConv3(x) * att_map

        out = self.fusion(torch.cat([x, y1, y2, y3], dim=1))

        # out = self.fusion(torch.add(out, x))

        # SF = self.SpatialAttention(x)
        #
        # branch1 = self.act(self.BN(self.branch1(x))) * SF
        # branch2 = self.act(self.BN(self.branch1(x))) * SF
        # branch3 = self.act(self.BN(self.branch1(x))) * SF
        #
        # out = self.act(self.BN(self.fusion(torch.cat([branch1, branch2, branch3], dim=1))))

        return out

class SPPFSAM(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()

        c_ = c1  # hidden channels
        self.ChannelAttention = ChannelAttention(c1, c_)
        self.SpatialAttention = SpatialAttention()

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):

        x = self.ChannelAttention(x)
        score = self.SpatialAttention(x)
        y1 = x * score

        res1 = self.m(x)
        y2 = res1 * score

        res2 = self.m(res1)
        y3 = res2 * score

        res3 = self.m(res2)
        y4 = res3 * score

        return self.cv2(torch.cat((y1, y2, y3, y4), dim=1))


class MultiScaleConv(nn.Module):
    def __init__(self, c1, c2):
        super(MultiScaleConv, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=3, padding=3),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=5, padding=5),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, dilation=7, padding=7),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )
    def forward(self, x):
        res = x

        y1 = self.Conv1(x)
        y2 = self.Conv2(x)
        y3 = self.Conv3(x)
        y4 = self.Conv4(x)

        return y1+y2+y3+y4+res


"""
--img 256 --batch 32 --epochs 150 --data ./data/coco.yaml --cfg ./models/yolov5m.yaml --weights ./models/yolov5m.pt
"""
