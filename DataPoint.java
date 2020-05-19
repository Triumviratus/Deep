/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deep;

import java.util.Arrays;

/**
 *
 * @author ARZavier
 */
public class DataPoint {
    private double[] fields;
    private double[] target;
    
    DataPoint(double[] fields, double[] target){
        this.fields = fields;
        this.target = target;
    }
    
    public double obtainFieldAt(int index){return this.fields[index];}
    public double[] obtainTarget(){return this.target;}
    public double[] obtainFields(){return this.fields;}
    public int obtainNumFeatures(){return fields.length;}
    public int obtainNumOutput(){return target.length;}
    @Override
    public String toString(){return "Fields: " + Arrays.toString(fields) + "\n\tTarget: " + Arrays.toString(target);}
}
