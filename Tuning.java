/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep;

import deep.Networks.Network;
import deep.Networks.SAE;
import deep.Functions.ActivationFunction;
import deep.Functions.Linear;
import deep.Functions.LogisticFunction;
import deep.Functions.SoftMax;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class Tuning {
    
    /**
     * Creates and tunes an SAE network
     * @param trainData
     * @param testData
     * @param name
     * @return
     */
    
    public static SAE tuneSAE(DataPoint[] trainData, DataPoint[] testData, String name) {
        ActivationFunction outputFunc = (trainData[0].obtainNumOutput() == 1) ? new Linear() : new SoftMax();
        int inputs = trainData[0].obtainNumFeatures();
        int outputs = trainData[0].obtainNumOutput();
        
        double[] learningRates = {0.1, 0.01, 0.001};
        
        SAE bestSAE = null;
        double best = Double.NEGATIVE_INFINITY; // Initialize the best state to be negative infinity
        
        File outputFile = Deep.createNewFile("Data/ExtraCredit/Tuning/" + name); // for output to csv
        String header = "metric, rate, ae, AE shape, FF Shape";
        Deep.appendToFile(header, outputFile);
        
        Random rand = new Random();
        for (double rate : learningRates) {
            // Search through the different rates specified
            for (int ae = 1; ae <= 3; ae++){
                // Enter Auto Encoder Sizes [1,3]
                for (int p = 0; p < 4; p++){
                    // Create 4 Different Networks
                    int[] shape = new int[ae];
                    int prev = inputs;
                    boolean check = true;
                    while(check) {
                        // Ascertain that we have exclusively decreasing order
                        prev = inputs;
                        for (int i = 0; i < shape.length; i++) {
                            // Generate random numbers for the nodes in the layer
                            int num = rand.nextInt(prev) + 1;
                            shape[i] = num;
                            prev = num;
                        }
                        // Ascertain that it is exclusively decreasing
                        prev = inputs;
                        check = false;
                        for (int value : shape) {
                            if (value >= prev)
                                check = true;
                            if (value == 0){
                                // Do not believe that this can happen
                                check = true;
                            }
                            prev = value;
                        }
                    }
                    double bestFF = Double.NEGATIVE_INFINITY; // The best feed forward network that sits on the model
                    SAE bestModel = null;
                    String ffString = "";
                    for (int ffLayer = 0; ffLayer < 3; ffLayer++) {
                        // Create ff networks with 0,1,2 hidden layers
                        for (int i = 0; i < 3; i++) {
                            // Utilize 3 random configurations for each network
                            int ffInputs = shape[shape.length - 1];
                            int[] ffShape = new int[ffLayer];
                            for (int j = 0; j < ffShape.length; j++){ffShape[j] = rand.nextInt(ffInputs) + 1;}
                            System.out.println("FF Shape: " + Arrays.toString(ffShape));
                            // Create the model
                            Network model = new Network(ffInputs, ffShape, outputs, new LogisticFunction(),
                                                        (outputs == 1) ? new Linear() : new SoftMax());
                            // Create the model
                            SAE test = new SAE(inputs, shape, model);
                            test.train(trainData, rate, 100);
                            double val = getMetric(outputs, test, testData);
                            if (val > bestFF) {
                                // Set the new model to this one because it is better
                                ffString = Arrays.toString(ffShape);
                                bestModel = test;
                                bestFF = val;
                            }
                            if (ffLayer == 0) break; // Only run once in the case of no hidden layers 
                        }
                    }
                    // Output to csv
                    String out = "" + bestFF + "," + rate + ", " + ae 
                            + "," + bestFF + "," + Arrays.toString(shape) + ", " + ffString;
                    Deep.appendToFile(out, outputFile);
                    if (bestFF > best) {
                        // We found a new global best network
                        System.out.println("\nBest");
                        best = bestFF;
                        bestSAE = bestModel;
                    }
                    System.out.println(out); // So we can see that it is still running because it takes forever
                }
            }
        }
        return bestSAE;
    }
    
    /**
     * Returns accuracy if classification
     * Returns squared error if regression
     * @param outputs
     * @param net
     * @param testData
     * @return 
     */
    
    private static double getMetric(int outputs, Network net, DataPoint[] testData){
        double value = 0;
        if (outputs == 1){
            // Regression (Utilize Squared Error)
            value = -Validator.squaredError(net, testData); // Subtract because we seek to maximize negative error
        } else {
            // Classification (Utilize Accuracy)
            value = Validator.accuracy(net, testData);
        }
        return value;
    }
}
