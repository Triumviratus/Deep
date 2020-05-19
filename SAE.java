/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep.Networks;

import deep.AEPoint;
import deep.DataPoint;
import deep.Functions.Linear;
import deep.Functions.LogisticFunction;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class SAE extends Network {
    
    AE[] stack; 
    Network model;
    int[] plan;
    
    /**
     * @param inputs
     * @param hiddenLayer
     * @param model we assume that the model has inputs (the last encoder in the stack)
     */
    
    public SAE(int inputs, int[] hiddenLayer, Network model) {
        super(inputs, new int[]{}, inputs, new LogisticFunction(), new Linear());
        // This creates a network with no hidden layers initially (just an input and output layer)
        network = new int[]{inputs}; // Causes the network object to not have the output layer
        plan = hiddenLayer; // The plan for the auto encoders
        stack = new AE[hiddenLayer.length]; // The auto encoders
        this.model = model; // The model that sits atop the atacked auto encoder
    }
    
    @Override
    public void train(DataPoint[] trainingData, double learningRate, int MaxIterations){
        trainAES(trainingData, learningRate, MaxIterations);
        /**
         * Now we have all the auto encoders in the stack. We now add the model to the top
         * of the encoder and utilize it. We will assume that the model has the same hidden
         * function as the encoder otherwise we cannot backprop through the SAE (just the model)
         */
        trainModel(trainingData, learningRate, MaxIterations);
    }
    
    /**
     * Trains the feed forward network atop the SAE
     * Runs backprop through the entire system
     * @param trainingData
     * @param learningRate
     * @param MaxIterations
     */
    
    private void trainModel(DataPoint[] trainingData, double learningRate, int MaxIterations) {
        this.outputFunction = model.outputFunction;
        
        // Expand the network
        int[] newNet = new int[network.length + model.network.length - 1];
        /**
         * Adds the SAE and the model together (-1 because we do not need the
         * input layer on the network as it is the same as the last layer in the SAE).
         */
        for (int i = 0; i < network.length; i++){newNet[i] = network[i];}
        for (int i = 0; i < model.network.length - 1; i++){newNet[i + network.length] = model.network[i + 1];}
        
        this.network = newNet;
        // Expand the weights matrix
        ArrayList<double[][]> weightList = new ArrayList<>(Arrays.asList(w.weights));
        for (double[][] cut : model.getWeights().weights){weightList.add(cut);}
        w.weights = weightList.toArray(new double[weightList.size()][][]);
        System.out.println("Final Shape");
        System.out.println(Arrays.toString(network));
        
        // Run backprop on entire system
        for (int i = 0; i < MaxIterations; i++){for (DataPoint d : trainingData){this.backprop(d, learningRate);}}
    }
    
    /**
     * Trains the auto encoders and builds the stack
     * @param trainingData
     * @param learningRate
     * @param MaxIterations
     */
    
    private void trainAES(DataPoint[] trainingData, double learningRate, int MaxIterations) {
        DataPoint[] aePoints = new DataPoint[trainingData.length];
        for (int i = 0; i < trainingData.length; i++){aePoints[i] = new AEPoint(trainingData[i].obtainFields());}
        
        // Creates the first auto encoder
        stack[0] = new AE(trainingData[0].obtainNumFeatures(), plan[0], hiddenFunction);
        stack[0].train(aePoints, learningRate, MaxIterations); // Runs gradient descent on the first encoder
        addEncoder(stack[0], true);
        
        // Adds the other encoders to the stack
        for (int i = 1; i < stack.length; i++) {
            stack[i] = new AE(plan[i-1], plan[i], hiddenFunction); // Creates the new auto encoder
            // Computes the outputs from the last layer in the net for each point
            DataPoint[] outputs = new DataPoint[aePoints.length];
            for (int j = 0; j < aePoints.length; j++){
                outputs[j] = new AEPoint(this.predict(aePoints[j]));
                System.out.println(outputs[i].toString());
            }
            stack[i].train(outputs, learningRate, MaxIterations); // Trains it with gradient descent
            addEncoder(stack[i], false); // Adds the encoder to the stack
        }
    }
    
    /**
     * Adds the hidden layer from the encoder and its trained weights to this.network
     * @param e
     * @param first 
     */
    private void addEncoder(AE e, boolean first){
        double[][] cut = e.cut();
        int[] newNet = new int[network.length + 1]; // Enlarges this.network to include the new encoder
        for (int i = 0; i < network.length; i++){newNet[i] = network[i];}
        // Adds the size of the new hidden layer
        newNet[newNet.length - 1] = cut[0].length;
        /**
         * The number of downstream nodes in the cut is equal to the size of the hidden layer in AE
         */
        network = newNet;
        System.out.println("Shape: " + Arrays.toString(plan));
        
        // Adds the new weight to the network
        if (!first) {
            ArrayList<double[][]> weightList = new ArrayList<>(Arrays.asList(w.weights));
            weightList.add(cut);
        } else {
            w.weights = new double[1][][];
            w.weights[0] = cut;
        }
        System.out.println("Weights: " + w.toString());
    }
    
    public void addModel(Network m){this.model = m;}
}
