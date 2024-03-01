'use strict';

import {buildConstantByNpy} from '../common/utils.js';

// SqueezeNet 1.0 fp16 model with 'nchw' input layout
export class SqueezeNetFP16Nchw {
  constructor() {
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = './weights/squeezenet1.0_fp16_nchw/';
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

  async buildConv_(input, name, options = {}) {
    const prefix = this.weightsUrl_ + name;
    const weights = await buildConstantByNpy(this.builder_,
        prefix + '_w_0.npy');
    const bias = await buildConstantByNpy(this.builder_, prefix + '_b_0.npy');
    options.bias = bias;
    options.activation = this.builder_.relu();
    return this.builder_.conv2d(input, weights, options);
  }

  async buildFire_(input, fireName) {
    const conv = await this.buildConv_(input, 'fire' + fireName + 'squeeze1x1');
    const conv1x1 = await this.buildConv_(conv,
        'fire' + fireName + 'expand1x1');
    const conv3x3 = await this.buildConv_(
        conv, 'fire' + fireName + 'expand3x3', {padding: [1, 1, 1, 1]});
    return this.builder_.concat([conv1x1, conv3x3], 1);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    const data = this.builder_.input('input', {dataType: this.inputOptions.
        dataType, dimensions: this.inputOptions.inputDimensions});
    const conv0 = await this.buildConv_(data, 'conv1', {strides: [2, 2]});
    const pool0 = this.builder_.maxPool2d(
        conv0, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire0 = await this.buildFire_(pool0, '2');
    const fire1 = await this.buildFire_(fire0, '3');
    const pool1 = this.builder_.maxPool2d(
        fire1, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire2 = await this.buildFire_(pool1, '4');
    const fire3 = await this.buildFire_(fire2, '5');
    const pool2 = this.builder_.maxPool2d(
        fire3, {windowDimensions: [3, 3], strides: [2, 2]});
    const fire4 = await this.buildFire_(pool2, '6');
    const fire5 = await this.buildFire_(fire4, '7');
    const fire6 = await this.buildFire_(fire5, '8');
    const fire7 = await this.buildFire_(fire6, '9');
    const conv25 = await this.buildConv_(fire7, 'conv10');
    const pool3 = this.builder_.averagePool2d(conv25);
    return this.builder_.reshape(pool3, [1, 1000]);
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
