// create a context and graph builder for 'gpu', 'cpu' or 'npu'.
const context =
    await navigator.ml.createContext({deviceType: 'gpu'});
const builder = new MLGraphBuilder(context);

// The following code builds a graph as:
// input1    ---+
//              +--- Mul ---> intermediateOutput1 ------+
// input1    ---+                                       |
//                                                      +--- Matmul---> output
//                                                      |
// constant1 -------------------------------------------+


// use tensors in 2 dimensions.
const TENSOR_DIMS = [2, 2];
const TENSOR_SIZE = 4;

// create MLOperandDescriptor object.
const desc = {dataType: 'float32', dimensions: TENSOR_DIMS, shape: TENSOR_DIMS};

// create constant1 which is a constant MLOperand with the value 0.5.
const constantBuffer1 = new Float32Array(TENSOR_SIZE).fill(0.5);
// ================================================================================
const constant1 = builder.constant(desc, constantBuffer1);

// create input1 which is one of the input MLOperands.
// Its value will be set before execution.
// ================================================================================
const input1 = builder.input('input1', desc);

// intermediateOutput1 is the output of the first Mul operation.
// ================================================================================
const intermediateOutput1 = builder.mul(input1, input1);

// output is the output MLOperand of the Matmul operation.
// ================================================================================
const output = builder.matmul(intermediateOutput1, constant1);

// Compile the constructed graph.
// ================================================================================
const graph = await builder.build({'output': output});

// Setup the input buffers with value 2.
const inputBuffer1 = new Float32Array(TENSOR_SIZE).fill(2);

desc.usage = MLTensorUsage.WRITE;
desc.writable = true;

const inputTensor1 = await context.createTensor(desc);
context.writeTensor(inputTensor1, inputBuffer1);

const outputTensor = await context.createTensor({
  ...desc,
  usage: MLTensorUsage.READ,
  readable: true,
  writable: false,
});

// Execute the compiled graph with the specified inputs.
// ================================================================================
context.dispatch(graph, {'input1': inputTensor1}, {'output': outputTensor});

// ================================================================================
const results = await context.readTensor(outputTensor);
console.log('Output value: ' + new Float32Array(results));
// Output value: 4, 4, 4, 4
