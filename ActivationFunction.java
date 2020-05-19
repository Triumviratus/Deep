/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep.Functions;

/**
 *
 * @author ARZavier
 */
public abstract class ActivationFunction {
    public ActivationFunction(){}
    public abstract double value(double weightedSum, double[] outputs);
    public abstract double derivative(double weightedSum);
}
