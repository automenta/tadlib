package com.codeberry.tadlib.provider.java;

public class TMutableArray {
    private volatile double[] data;
    public final Shape shape;

    public TMutableArray(Shape shape) {
        this(new double[shape.size], shape);
    }

    public TMutableArray(double[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
    }

    /**
     * @return 0 when out of bounds
     */
    double dataAt(int... indices) {
        return shape.isValid(indices) ? data[shape.calcDataIndex(indices)] : 0;
    }

    public void setAt(int[] indices, double v) {
        setAtOffset(shape.calcDataIndex(indices), v);
    }

    void setAtOffset(int offset, double v) {
        data[offset] = v;
    }

    public double[] getData() {
        return data;
    }

    /**
     * The current instance cannot be used after this call.
     */
    synchronized NDArray migrateToImmutable() {
        NDArray immutable = new NDArray(this.data, shape);

        this.data = null;

        return immutable;
    }
}
