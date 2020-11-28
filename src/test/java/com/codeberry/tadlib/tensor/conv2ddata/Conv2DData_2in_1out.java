package com.codeberry.tadlib.tensor.conv2ddata;

public class Conv2DData_2in_1out {
    // Dims: 1, 5, 5, 2,
    public static final double[][][][] input = new double[][][][] {{{{ 0.,  1.},
            { 2.,  3.},
            { 4.,  5.},
            { 6.,  7.},
            { 8.,  9.}},

            {{10., 11.},
                    {12., 13.},
                    {14., 15.},
                    {16., 17.},
                    {18., 19.}},

            {{20., 21.},
                    {22., 23.},
                    {24., 25.},
                    {26., 27.},
                    {28., 29.}},

            {{30., 31.},
                    {32., 33.},
                    {34., 35.},
                    {36., 37.},
                    {38., 39.}},

            {{40., 41.},
                    {42., 43.},
                    {44., 45.},
                    {46., 47.},
                    {48., 49.}}}};
    // Dims: 3, 3, 2, 1,
    public static final double[][][][] filter = new double[][][][] {{{{ 0.},
            { 1.}},

            {{ 2.},
                    { 3.}},

            {{ 4.},
                    { 5.}}},


            {{{ 6.},
                    { 7.}},

                    {{ 8.},
                            { 9.}},

                    {{10.},
                            {11.}}},


            {{{12.},
                    {13.}},

                    {{14.},
                            {15.}},

                    {{16.},
                            {17.}}}};
    // Dims: 1, 5, 5, 1,
    public static final double[][][][] y = new double[][][][] {{{{ 780.},
            {1250.},
            {1526.},
            {1802.},
            {1180.}},

            {{1806.},
                    {2685.},
                    {2991.},
                    {3297.},
                    {2070.}},

            {{2946.},
                    {4215.},
                    {4521.},
                    {4827.},
                    {2970.}},

            {{4086.},
                    {5745.},
                    {6051.},
                    {6357.},
                    {3870.}},

            {{2028.},
                    {2690.},
                    {2822.},
                    {2954.},
                    {1660.}}}};
    // Dims: 1, 5, 5, 1,
    public static final double[][][][] grad_y = new double[][][][] {{{{ 0.},
            { 1.},
            { 2.},
            { 3.},
            { 4.}},

            {{ 5.},
                    { 6.},
                    { 7.},
                    { 8.},
                    { 9.}},

            {{10.},
                    {11.},
                    {12.},
                    {13.},
                    {14.}},

            {{15.},
                    {16.},
                    {17.},
                    {18.},
                    {19.}},

            {{20.},
                    {21.},
                    {22.},
                    {23.},
                    {24.}}}};
    // Dims: 1, 5, 5, 2,
    public static final double[][][][] grad_input = new double[][][][] {{{{  16.,   28.},
            {  52.,   73.},
            {  82.,  109.},
            { 112.,  145.},
            { 112.,  136.}},

            {{ 108.,  141.},
                    { 240.,  294.},
                    { 312.,  375.},
                    { 384.,  456.},
                    { 336.,  387.}},

            {{ 318.,  381.},
                    { 600.,  699.},
                    { 672.,  780.},
                    { 744.,  861.},
                    { 606.,  687.}},

            {{ 528.,  621.},
                    { 960., 1104.},
                    {1032., 1185.},
                    {1104., 1266.},
                    { 876.,  987.}},

            {{ 688.,  760.},
                    {1168., 1279.},
                    {1234., 1351.},
                    {1300., 1423.},
                    { 976., 1060.}}}};
    // Dims: 3, 3, 2, 1,
    public static final double[][][][] grad_filter = new double[][][][] {{{{ 5360.},
            { 5600.}},

            {{ 6840.},
                    { 7130.}},

            {{ 5520.},
                    { 5744.}}},


            {{{ 7800.},
                    { 8050.}},

                    {{ 9800.},
                            {10100.}},

                    {{ 7800.},
                            { 8030.}}},


            {{{ 5520.},
                    { 5680.}},

                    {{ 6840.},
                            { 7030.}},

                    {{ 5360.},
                            { 5504.}}}};
}
