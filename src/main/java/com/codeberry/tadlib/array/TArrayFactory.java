package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.java.JavaIntArray;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;

import java.util.Random;

import static com.codeberry.tadlib.provider.ProviderStore.*;

public abstract class TArrayFactory {

    public static NDArray onesShaped(int... dims) {
        return ones(shape(dims));
    }

    public static NDArray ones(Shape shape) {
        return arrayFillWith(shape.normalOrderedCopy(), 1);
    }

    public static NDArray zerosShaped(int... dims) {
        return zeros(shape(dims));
    }

    public static NDArray zeros(Shape shape) {
        return arrayFillWith(shape.normalOrderedCopy(), 0.0);
    }

    public static JavaIntArray intZerosShaped(int... dims) {
        return intZeros(shape(dims));
    }

    public static JavaIntArray intZeros(Shape shape) {
        return intArrayFillWith(shape, 0);
    }

    public static NDArray randomInt(Random rand, int from, int to, int size) {
        return array(randomIntDoubles(rand, from, to, size));
    }

    private static double[] randomIntDoubles(Random rand, int from, int to, int len) {
        double[] row = new double[len];
        int diff = to - from;
        for (int i = 0; i < row.length; i++) {
            row[i] = from + rand.nextInt(diff);
        }
        return row;
    }

    public static NDArray random(Random rand, int size) {
        return array(randomDoubles(rand, size));
    }

    public static NDArray randomWeight(Random rand, Shape shape) {
        return randomWeight(rand, Math.toIntExact(shape.size)).reshape(shape);
//        return DisposalRegister.disposeAllExceptReturnedValue(() ->
//                randomWeight(rand, size).reshape(shape));
    }

    public static NDArray randomWeight(Random rand, int size) {
        return array(randomDoubles(rand, size)).add(-0.5).mul(2.0).mul(Math.sqrt(2.0 / size));
    }

    static double[] randomDoubles(Random rand, int len) {
        double[] row = new double[len];
        for (int i = 0; i < row.length; i++) {
            row[i] = rand.nextDouble();
        }
        return row;
    }

    public static NDArray range(int count) {
        return array(rangeDoubles(count));
    }

    public static JavaIntArray intRange(int count) {
        return array(rangeInt(count));
    }

    public static double[] rangeDoubles(int count) {
        double[] data = new double[count];
        for (int i = 0; i < count; i++)
            data[i] = i;
        return data;
    }

    public static int[] rangeInt(int count) {
        int[] data = new int[count];
        for (int i = 0; i < count; i++)
            data[i] = i;
        return data;
    }

    public static NDArray fillLike(Shape shape, NDArray zeroDim) {
        if (zeroDim.shape.dimCount != 0)
            throw new IllegalArgumentException("value must be of zero dim");

        return arrayFillWith(shape.normalOrderedCopy(), (double) zeroDim.toDoubles());
    }

}
