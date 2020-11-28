package com.codeberry.tadlib.array;

import org.junit.jupiter.api.Test;

import static java.util.Arrays.deepEquals;
import static org.junit.jupiter.api.Assertions.*;

class TArrayReshape {
    @Test
    public void reshape() {
        TArray m = new TArray(new double[]{
                0, 1, 2, 3, 4, 5
        });
        double[] org = (double[]) m.toDoubles();
        assertArrayEquals(new double[] {0, 1, 2, 3, 4, 5}, org);

        TArray out_1x6 = m.reshape(1, 6);
        double[][] vals_1x6 = (double[][]) out_1x6.toDoubles();
        assertArrayEquals(new double[] {0, 1, 2, 3, 4, 5}, vals_1x6[0]);

        TArray out_2x3 = m.reshape(2, 3);
        double[][] vals_2x3 = (double[][]) out_2x3.toDoubles();
        assertArrayEquals(new double[] {0, 1, 2}, vals_2x3[0]);
        assertArrayEquals(new double[] {3, 4, 5}, vals_2x3[1]);

        TArray out_2x1x3 = m.reshape(2, 1, 3);
        double[][][] vals_2x1x3 = (double[][][]) out_2x1x3.toDoubles();
        assertTrue(deepEquals(new double[][] {{0, 1, 2}},
                vals_2x1x3[0]));
        assertTrue(deepEquals(new double[][] {{3, 4, 5}},
                vals_2x1x3[1]));

        TArray out_3x2 = m.reshape(3, 2);
        double[][] vals_3x2 = (double[][]) out_3x2.toDoubles();
        assertArrayEquals(new double[] {0, 1}, vals_3x2[0]);
        assertArrayEquals(new double[] {2, 3}, vals_3x2[1]);
        assertArrayEquals(new double[] {4, 5}, vals_3x2[2]);

        TArray out_6x1 = m.reshape(6, 1);
        double[][] vals_6x1 = (double[][]) out_6x1.toDoubles();
        assertArrayEquals(new double[] {0}, vals_6x1[0]);
        assertArrayEquals(new double[] {1}, vals_6x1[1]);
        assertArrayEquals(new double[] {2}, vals_6x1[2]);
        assertArrayEquals(new double[] {3}, vals_6x1[3]);
        assertArrayEquals(new double[] {4}, vals_6x1[4]);
        assertArrayEquals(new double[] {5}, vals_6x1[5]);
    }

    @Test
    public void reshape_Invalid_Shapes() {
        TArray m = new TArray(new double[]{
                0, 1, 2, 3, 4, 5
        });
        assertDoesNotThrow(() -> m.reshape(1, 1, 1, 6));

        assertThrows(TArray.InvalidTargetShape.class,
                () -> m.reshape(7));
        assertThrows(TArray.InvalidTargetShape.class,
                () -> m.reshape(2,4));
    }

}