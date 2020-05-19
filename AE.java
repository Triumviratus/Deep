/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep.Networks;

import deep.Functions.ActivationFunction;
import deep.Functions.Linear;
import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class AE extends Network {
    public AE(int inputs, int hiddenLayer, ActivationFunction hiddenFunc) {
        super(inputs, new int[]{hiddenLayer}, inputs, hiddenFunc, new Linear());
        if (w.weights.length != 2){
            // We have more than 3 layers, so something is broken
            System.out.println("Inadequate Auto Encoder");
        }
        System.out.println("Auto Encoder Shape: " + Arrays.toString(network));
    }
    
    /**
     * Cuts off the output layer of the network
     * @return the weights from the input layer to the hidden layer
     */
    
    public double[][] cut(){return w.weights[0];}
}
