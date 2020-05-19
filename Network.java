/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep.Networks;

import deep.*;
import deep.Functions.*;
/**
 *
 * @author ARZavier
 */
public class Network {
    
    int[] network;
    Weights w;
    ActivationFunction hiddenFunction;
    ActivationFunction outputFunction;
    
    /**
     * Builds a multi-layer feed forward network
     * @param inputs the number of inputs into the network
     * @param hiddenLayer the shape of the hidden layers in the network
     * @param outputs the outputs from the network
     * @param hiddenFunc the Activation Function on the hidden nodes
     * @param outputFunc the Activation Function on the output layer
     */
    public Network (int inputs, int[] hiddenLayer, int outputs, ActivationFunction hiddenFunc, ActivationFunction outputFunc) {
        network = new int[hiddenLayer.length + 2];
        network[0] = inputs;
        for (int i = 0; i < hiddenLayer.length; i++){network[i + 1] = hiddenLayer[i];}
        network[network.length - 1] = outputs;
        w = new Weights(network); // Initializes Weight Array
        w.randomizeWeights();
        hiddenFunction = hiddenFunc;
        outputFunction = outputFunc;
    }
    
    /**
     * Creates an output of each node in the network
     * The output is the weighted sum of the upstream nodes times their respective weights
     * This is the dot product of the inputs and the weight vectors
     * @param d data point that is an input to the network
     * @return an output of every node in the network
     */
    
    double[][] genOutput (DataPoint d){
        double[][] outputs = blankNodes();
        for (int i = 0; i < outputs[0].length; i++){
            // Feed the data point into the network
            outputs[0][i] = d.obtainFieldAt(i);
        }
        for (int layer = 1; layer < outputs.length; layer++) {
            for (int node = 0; node < outputs[layer].length; node++) {
                // Computes the dot product of inputs with weights
                double sum = 0;
                for (int i = 0; i < outputs[layer - 1].length; i++) {
                    /**
                     * It is not the first layer past the inputs, so we need 
                     * to utilize the activation from the previous layer.
                     */
                    if (layer > 1)
                        sum += hiddenFunction.value(outputs[layer - 1][i], new double[]{}) * w.obtainWeight(layer - 1, i, node);
                    else
                        sum += outputs[layer - 1][i] * w.obtainWeight(layer - 1, i, node);
                }
                outputs[layer][node] = sum;
            }
        }
        return outputs; // Returns the output from the network
    }
    
    /**
     * Creates the output from the network by applying the output function to
     * the output layer from GenOutput
     * @param d data point that is the input to the network
     * @return the output from the network
     */
    
    public double[] predict(DataPoint d) {
        
        double[] weightedOut = this.genOutput(d)[network.length-1];
        double[] outs = new double[weightedOut.length];
        for (int i = 0; i < weightedOut.length; i++){outs[i] = outputFunction.value(weightedOut[i], weightedOut);}
        // Apply Output Function
        return outs;
    }
    
    /**
     * Trains the network by utilizing maxIterations of backprop
     * @param trainingData data to be trained on
     * @param trainingRate learning rate
     * @param maxIterations number of iterations
     */
    
    public void train(DataPoint[] trainingData, double trainingRate, int maxIterations) {
        int numOfIterations = 0;
        boolean shouldContinue = true;
        
        while (shouldContinue) {
            // Loop through our training data
            for (DataPoint currPoint : trainingData){backprop(currPoint, trainingRate);}
            numOfIterations++;
            if (numOfIterations > maxIterations){
                // Termination
                shouldContinue = false;
            }
        }
    }
    
    /**
     * @param d the current training data point
     * @param learningRate
     */
    
    protected void backprop(DataPoint d, double learningRate) {
        double[] target = d.obtainTarget();
        double[][] entireOutput = genOutput(d);
        Weights deltas = new Weights(this.network);
        
        // Gradient descent for the output layers
        int upstreamLayerIndex = getIndexOfOutputLayer() -1;
        // Loop though the output nodes
        for (int outputIndex = 0; outputIndex < this.getNumberOfOutputs(); outputIndex++){
            double error = target[outputIndex] - 
                    outputFunction.value(entireOutput[this.getIndexOfOutputLayer()][outputIndex], 
                            entireOutput[this.getIndexOfOutputLayer()]);
            // loop for the last hidden layer nodes
            for (int upstreamIndex = 0; upstreamIndex < this.getNumberOfNodesAtLayer(upstreamLayerIndex); upstreamIndex++) {
                double derivative = hiddenFunction.derivative(entireOutput[upstreamLayerIndex][upstreamIndex]);
                double change = -error * derivative;
                deltas.setWeight(change, getIndexOfOutputLayer() - 1, upstreamIndex, outputIndex);
            }
        }
        
        // Start at First Hidden Layer and Move Back
        for (int layer = getIndexOfOutputLayer() - 1; layer > 0; layer--){
            // Backprop for all other layers
            
            // Loop through the nodes of the hidden layer
            for (int hiddenIndex = 0; hiddenIndex < getNumberOfNodesAtLayer(layer); hiddenIndex++){
                // Loop through upstream nodes
                
                // Generate the sum of the deltas to downstream nodes
                double sum = 0.0;
                for (int downstreamIndex = 0; downstreamIndex < getNumberOfNodesAtLayer(layer + 1); downstreamIndex++) {
                    // Add up changes to all the weights in downstream layer
                    sum += deltas.obtainWeight(layer, hiddenIndex, downstreamIndex) 
                            * w.obtainWeight(layer, hiddenIndex, downstreamIndex);
                }
                for (int upstreamIndex = 0; upstreamIndex < getNumberOfNodesAtLayer(layer - 1); upstreamIndex++) {
                    // Set Weight Deltas
                    double change = hiddenFunction.derivative(entireOutput[layer - 1][upstreamIndex] * sum);
                    deltas.setWeight(change, layer - 1, upstreamIndex, hiddenIndex);
                }
            }
        }
        // Change the Weights
        for (int layer = 0; layer < network.length - 1; layer++) {
            for (int upstream = 0; upstream < getNumberOfNodesAtLayer(layer); upstream++) {
                for (int downstream = 0; downstream < getNumberOfNodesAtLayer(layer + 1); downstream++) {
                    double dW = -learningRate * deltas.obtainWeight(layer, upstream, downstream) 
                            * hiddenFunction.value(entireOutput[layer][upstream], new double[]{});
                    w.changeWeight(dW, layer, upstream, downstream);
                }
            }
        }
    }
    
    /**
     * Creates a double array to store the output values from the network
     * @return
     */
    
    private double[][] blankNodes() {
        double[][] outputs = new double[network.length][];
        for (int i = 0; i < network.length; i++){outputs[i] = new double[network[i]];}
        return outputs;
    }
    
    public int getNumberOfOutputs(){return this.network[this.network.length - 1];}
    public int getNumberOfInputs(){return this.network[0];}
    
    public int getIndexOfOutputLayer(){return this.network.length - 1;}
    public int getNumberOfNodesAtLayer(int layer){return this.network[layer];}
    
    public Weights getWeights(){return w;}
    
    public ActivationFunction getHiddenFunction(){return hiddenFunction;}
    public ActivationFunction getOutputFunction(){return outputFunction;}
    
}
