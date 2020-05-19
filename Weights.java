/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep.Networks;

import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author ARZavier
 */
public class Weights {
    
    protected double[][][] weights;
    /**
     * Creates of the copy of the weights object w
     * @param w weights to be copied
     */
    public Weights(Weights w){
        // Copies the weights from w into this
        weights = new double[w.weights.length][][];
        for (int layer = 0; layer < w.weights.length; layer++) {
            weights[layer] = new double[w.weights[layer].length][];
            for (int upstream = 0; upstream < w.weights[layer].length; upstream++) {
                weights[layer][upstream] = new double[w.weights[layer][upstream].length];
                for (int downstream = 0; downstream < w.weights[layer][upstream].length; downstream++) {
                    weights[layer][upstream][downstream] = w.weights[layer][upstream][downstream];
                }
            }
        }
    }
    
    /**
     * Stores the weights on a feed forward neural network
     * @param nodesPerLayer 
     */
    public Weights(int[] nodesPerLayer) {
        this.weights = new double[nodesPerLayer.length - 1][][];
        
        for (int layer = 0; layer < nodesPerLayer.length - 1; layer++){
            this.weights[layer] = new double[nodesPerLayer[layer]][nodesPerLayer[layer + 1]];
        }
    }
    
    public double obtainWeight(int layer, int upstreamIndex, int downstreamIndex) {
        return this.weights[layer][upstreamIndex][downstreamIndex];
    }
    
    public void setWeight(double newWeight, int layer, int upstreamIndex, int downstreamIndex){
        this.weights[layer][upstreamIndex][downstreamIndex] = newWeight;
    }
    
    public void changeWeight(double change, int layer, int upstreamIndex, int downstreamIndex) {
        this.weights[layer][upstreamIndex][downstreamIndex] += change;
    }
    
    public int obtainNumberOfLayers(){return this.weights.length;}
    
    /**
     * Obtains the number of nodes in a given layer
     * @param layer the upstream layer of the edge
     * @return the number of nodes
     */
    
    public int obtainNodesInLayer(int layer){
        if (layer == weights.length){
            // Is Output Layer
            return this.weights[layer-1][0].length;
        }
        return this.weights[layer].length;
    }
    
    /**
     * Fills the weights with some constant value
     * @param x
     */
    
    public void fill(double x) {
        for (int layer = 0; layer < weights.length; layer++){
            for (int upstream = 0; upstream < weights[layer].length; upstream++){
                Arrays.fill(weights[layer][upstream], x);
            }
        }
    }
    
    /**
     * Randomize the weights
     */
    
    public void randomizeWeights(){
        Random rand = new Random();
        for (int layer = 0; layer < weights.length; layer++){
            for (int upstream = 0; upstream < weights[layer].length; upstream++) {
                for (int downstream = 0; downstream < weights[layer][upstream].length; downstream++){
                    weights[layer][upstream][downstream] = rand.nextDouble();
                }
            }
        }
    }
    
    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        for (int layer = 0; layer < weights.length; layer++) {
            sb.append ("Layer: ").append(layer).append('\n');
            for (int upstream = 0; upstream < weights[layer].length; upstream++) {
                sb.append("\tUpstream: ").append(upstream).append('\n');
                for (int downstream = 0; downstream < weights[layer][upstream].length; downstream++) {
                    sb.append("\t\tDownstream: ").append(downstream).
                            append(": ").append(weights[layer][upstream][downstream]).append("\n");
                }
            }
        }
        return sb.toString();
    }
    
    /**
     * Adds the two weight vectors
     * @param a
     * @param b
     * @param scalar a scalar multiple to multiply the sum
     * @return a new weights vector
     */
    
    public static Weights add(Weights a, Weights b, double scalar) {
        Weights s = new Weights(a);
        for (int layer = 0; layer < s.weights.length; layer++) {
            // Goes through each weight and adds them
            for (int upstream = 0; upstream < s.weights[layer][upstream].length; upstream++){
                for (int downstream = 0; downstream < s.weights[layer][upstream].length; downstream++){
                    s.weights[layer][upstream][downstream] = scalar * 
                            (a.weights[layer][upstream][downstream] + b.weights[layer][upstream][downstream]);
                }
            }
        }
        return s;
    }
    
    /**
     * Subtracts b from a (i.e., a - b)
     * scalar * (a-b)
     * @param a
     * @param b
     * @param scalar the constant value to multiply the differences
     * @return
     */
    
    public static Weights sub(Weights a, Weights b, double scalar) {
        Weights s = new Weights(a);
        for (int layer = 0; layer < s.weights.length; layer++) {
            // Goes through each weight and subtracts them
            for (int upstream = 0; upstream < s.weights[layer][upstream].length; upstream++){
                for (int downstream = 0; downstream < s.weights[layer][upstream].length; downstream++){
                    s.weights[layer][upstream][downstream] = scalar * 
                            (a.weights[layer][upstream][downstream] - b.weights[layer][upstream][downstream]);
                }
            }
        }
        return s;
    }
    
}
