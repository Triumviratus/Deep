/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep;

/**
 *
 * @author ARZavier
 */
public class AEPoint extends DataPoint {
    /**
     * Creates a data point whose target is itself
     * @param fields
     */
    public AEPoint(double[] fields){super (fields, fields);}
}
