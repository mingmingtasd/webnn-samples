'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// ResNet50 V1 fp16 model with 'nchw' input layout
export class ResNet50V1FP16Nchw {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = './weights/resnet50v1_fp16_nchw_optimized/';
    this.inputOptions = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      norm: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels1000.txt',
      inputDimensions: [1, 3, 224, 224],
      dataType: 'float16',
    };
    this.outputDimensions = [1, 1000];
  }

  async buildConv_(input, name, stageName, relu, options = undefined) {
    let prefix = '';
    if (stageName !== '') {
      prefix = this.weightsUrl_ + 'stage' + stageName + '_conv' +
          name;
    } else {
      prefix = this.weightsUrl_ + 'conv' + name;
    }
    const weight = await buildConstantByNpy(this.builder_, prefix + '_w.npy');
    options.bias = await buildConstantByNpy(this.builder_, prefix + '_b.npy');
    if (relu) {
      options.activation = this.builder_.relu();
    }

    return this.builder_.conv2d(input, weight, options);
  }

  async buildGemm_(input, name) {
    const prefix = this.weightsUrl_ + 'dense' + name;
    const weightName = prefix + '_w.npy';
    const weight = await buildConstantByNpy(this.builder_, weightName);
    const biasName = prefix + '_b.npy';
    const bias = await buildConstantByNpy(this.builder_, biasName);
    const options =
        {c: this.builder_.reshape(bias, [1, 1000]), bTranspose: true};
    return this.builder_.gemm(input, weight, options);
  }

  async buildBottleneck_(
      input, stageName, nameIndex, downsample = false, stride = 1) {
    let residual = input;
    let strides = [1, 1];
    if (downsample) {
      strides = [stride, stride];
    }
    const conv1 = await this.buildConv_(input, nameIndex,
        stageName, true, {strides});
    const conv2 = await this.buildConv_(conv1, parseInt(nameIndex) + 1,
        stageName, true, {padding: [1, 1, 1, 1]});
    const conv3 = await this.buildConv_(conv2,
        parseInt(nameIndex) + 2, stageName, false, {});
    if (downsample) {
      residual = await this.buildConv_(
          input, parseInt(nameIndex) + 3, stageName, false, {strides});
    }
    const add = await this.builder_.add(conv3, residual);
    return this.builder_.relu(add);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const data = this.builder_.input('input', {dataType:
    this.inputOptions.dataType,
    dimensions: this.inputOptions.inputDimensions});
    const conv1 = await this.buildConv_(
        data, '0', '', true, {padding: [3, 3, 3, 3], strides: [2, 2]});
    const pool1 = await this.builder_.maxPool2d(conv1,
        {windowDimensions: [3, 3], padding: [1, 1, 1, 1], strides: [2, 2]});

    // Stage 1
    const bottleneck1 = await this.buildBottleneck_(pool1, '1', '0', true);
    const bottleneck2 = await this.buildBottleneck_(bottleneck1, '1', '4');
    const bottleneck3 = await this.buildBottleneck_(bottleneck2, '1', '7');

    // Stage 2
    const bottleneck4 = await this.buildBottleneck_(bottleneck3, '2', '0',
        true, 2);
    const bottleneck5 = await this.buildBottleneck_(bottleneck4, '2', '4');
    const bottleneck6 = await this.buildBottleneck_(bottleneck5, '2', '7');
    const bottleneck7 = await this.buildBottleneck_(bottleneck6, '2', '10');

    // Stage 3
    const bottleneck8 = await this.buildBottleneck_(bottleneck7, '3', '0',
        true, 2);
    const bottleneck9 = await this.buildBottleneck_(bottleneck8, '3', '4');
    const bottleneck10 = await this.buildBottleneck_(bottleneck9, '3', '7');
    const bottleneck11 = await this.buildBottleneck_(bottleneck10, '3', '10');
    const bottleneck12 = await this.buildBottleneck_(bottleneck11, '3', '13');
    const bottleneck13 = await this.buildBottleneck_(bottleneck12, '3', '16');

    // Stage 4
    const bottleneck14 = await this.buildBottleneck_(bottleneck13, '4', '0',
        true, 2);
    const bottleneck15 = await this.buildBottleneck_(bottleneck14, '4', '4');
    const bottleneck16 = await this.buildBottleneck_(bottleneck15, '4', '7');

    const pool2 = this.builder_.averagePool2d(bottleneck16);
    const reshape = this.builder_.reshape(pool2, [1, 2048]);
    return this.buildGemm_(reshape, '0');
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({'output': outputOperand});
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if (this.graph_ !== null && 'dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  async compute(inputBuffer, outputBuffer) {
    const inputs = {'input': inputBuffer};
    const outputs = {'output': outputBuffer};
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results;
  }
}
