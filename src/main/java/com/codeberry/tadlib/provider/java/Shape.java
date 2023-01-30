package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.exception.InvalidTargetShape;
import com.codeberry.tadlib.provider.ProviderStore;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;

import static java.lang.Math.min;

public class Shape {
    protected final int[] dims;
    public final int size, dimCount;

    private Shape() {
        //this(new int[]{});
        this.dims = new int[] { };
        this.dimCount = 0;
        this.size = 1;
    }

    public Shape(int[] dims) {
        if (dims.length==0)
            throw new UnsupportedOperationException("use zeroDim");
        this.dims = dims;
        this.dimCount = dims.length;
        this.size = mul(dims);
    }

    public static final Shape zeroDim = new Shape();

    public static Shape shape(int... dims) {
        return dims.length == 0 ? zeroDim : new Shape(dims);
    }

    public static String asString(Shape shape) {
        if (shape.dimCount == 0)
            return "<>";

        StringBuilder buf = new StringBuilder("<");
        for (int i = 0; i < shape.dimCount; i++) {
            buf.append(shape.at(i)).append(",");
        }
        buf.setLength(buf.length() - 1);
        buf.append(">");
        return buf.toString();
    }

//    public static JavaShape like(double[][] values) {
//        return new JavaShape(values.length, values[0].length);
//    }

    public int at(int dim) {
        return dims.length == 0 ? 0 : dims[wrapIndex(dim)];
    }

    int wrapIndex(int dim) {
        return dim >= 0 ? dim : dimCount + dim;
    }

    public Shape reshape(int... dims) {
        return new Shape(reshapeDims(dims));
    }

    private static int mul(int[] dims) {
        int targetSize = 1;
        for (int dim : dims) {
            targetSize *= dim;
        }
        return targetSize;
    }

    int getBroadcastOffset(int[] indices) {
        int offset = 0;
        int blockSize = 1;
        int dimCount = -this.dimCount;
        for (int i = -1; i >= dimCount; i--) {
            int shapeDimSize = at(i);
            offset += blockSize * min(indices[indices.length + i], shapeDimSize - 1);
            blockSize *= shapeDimSize;
        }
        return offset;
    }

//    public Shape squeeze(int... removeSingleDimsIndices) {
//        if (this instanceof ReorderedJavaShape) {
//            throw new UnsupportedOperationException("Reordered not supported");
//        }
//        int[] _tmp = Arrays.copyOf(removeSingleDimsIndices, removeSingleDimsIndices.length);
//        for (int i = 0; i < _tmp.length; i++) {
//            if (_tmp[i] <= -1) {
//                _tmp[i] += dimCount;
//            }
//        }
//        BitSet bitSet = new BitSet();
//        for (int idx : _tmp) {
//            if (!bitSet.get(idx))
//                bitSet.set(idx);
//            else
//                throw new DuplicatedSqueezeDimension("dim=" + idx);
//        }
//        for (int i = 0; i < _tmp.length; i++) {
//            if (this.dims[_tmp[i]] != 1)
//                throw new CannotSqueezeNoneSingleDimension("index=" + i + " dim=" + _tmp[i]);
//        }
//        int[] dims = new int[dimCount - _tmp.length];
//        int idx = 0;
//        for (int i = 0; i < this.dims.length; i++) {
//            if (!bitSet.get(i)) {
//                dims[idx] = this.dims[i];
//                idx++;
//            }
//        }
//        return new Shape(dims);
//    }

    // remove any reordering etc.
    public Shape normalOrderedCopy() {
        return Shape.shape(dims.clone());
    }

//    public int[] getDimensions() {
//        return dims;
//    }

    public double[] convertDataToShape(double[] data, Shape tgtShape) {
        if (tgtShape.getClass() == this.getClass())
            return data.clone();//Arrays.copyOf(data, data.length);

        double[] cp = new double[data.length];

        fillIntoDataArray(data, cp, tgtShape, newIndexArray(), 0);

        return cp;
    }

    private void fillIntoDataArray(double[] src, double[] tgt,
                                   Shape tgtShape, int[] indices, int dim) {
        int len = at(dim);
        if (dim == dimCount - 1) {
            //...last index
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                tgt[tgtShape.calcDataIndex(indices)] = src[calcDataIndex(indices)];
            }
        } else {
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                fillIntoDataArray(src, tgt, tgtShape, indices, dim + 1);
            }
        }
    }

    @Override
    public String toString() {
        return Shape.asString(this);
    }

    public Shape copy() {
        return this;
    }

    public int getDimCount() {
        return dimCount;
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Shape os && equalsShape(os);
    }

    @Override
    public int hashCode() {
        return shapeHashCode();
    }

    public int atOrDefault(int dim, int defaultValue) {
        int dimCount = this.dimCount;
        int i = (dim >= 0 ? dim : dimCount + dim);
        return i >= 0 && i < dimCount ? at(i) : defaultValue;
    }

    public int calcDataIndex(int[] indices) {
        int idx = 0;
        int blockSize = 1;
        for (int i = indices.length - 1; i >= 0; i--) {
            idx += indices[i] * blockSize;
            blockSize *= at(i);
        }
        return idx;
    }

    public int[] newIndexArray() {
        return new int[dimCount];
    }

    public Object newValueArray(Class<?> componentType) {
        return Array.newInstance(componentType, toDimArray());
    }

    public boolean[] newCollapseArray() {
        //Arrays.fill(b, FALSE);
        return new boolean[dimCount];
    }

    public int[] toDimArray() {
        int[] dims = new int[dimCount];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = at(i);
        }
        return dims;
    }

    public boolean equalsShape(Shape other) {
        if (this == other) return true;
        if (other != null) {
            if (size == other.size) {
                if (dimCount == other.dimCount) {
                    for (int i = dimCount - 1; i >= 0; i--) {
                        if (at(i) != other.at(i)) {
                            return false;
                        }
                    }
                    return true;
                }
            }
        }
        return false;
    }

    public int shapeHashCode() {
        int result = Objects.hash((long) size, dimCount);
        for (int i = dimCount - 1; i >= 0; i--) {
            result = 31 * result + at(i);
        }
        return result;
    }

    public int[] reshapeDims(int... dims) {
        long restSize = size;
        int deduceDim = -1;

        for (int i = 0; i < dims.length; i++) {
            int dim = dims[i];
            if (dim > 0) {
                if ((restSize % dim) != 0) {
                    throw new InvalidTargetShape("Not divisible: Index: " + i + " dim: " + dim);
                }
                restSize /= dim;
            } else if (dim == -1) {
                if (deduceDim != -1) {
                    throw new InvalidTargetShape("Multiple -1 dim not allowed: Index: " + i);
                }
                deduceDim = i;
            } else {
                throw new InvalidTargetShape("Invalid dim: Index: " + i + " dim:" + dim);
            }
        }

        if (deduceDim != -1) {
            dims[deduceDim] = Math.toIntExact(restSize);
        }

        return dims;
    }

    public boolean[] forEachDim(Predicate<Integer> mapper, IntFunction<boolean[]> returnFactory) {
        boolean[] data = returnFactory.apply(dimCount);
        for (int i = 0; i < data.length; i++) {
            data[i] = mapper.test(at(i));
        }
        return data;
    }

    public <A> A[] forEachDim(Function<Integer, A> mapper, IntFunction<A[]> returnFactory) {
        A[] data = returnFactory.apply(dimCount);
        for (int i = 0; i < data.length; i++) {
            data[i] = mapper.apply(at(i));
        }
        return data;
    }

    public boolean hasSingleDims() {
        for (int i = 0; i < dimCount; i++)
            if (at(i) == 1)
                return true;
        return false;
    }

    public boolean correspondsTo(Shape other) {
        int dimCount = this.dimCount;

        if (dimCount == other.dimCount) {
            for (int i = 0; i < dimCount; i++) {
                if (at(i) != other.at(i)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    public Shape withDimAt(int index, int dimLength) {
        int[] dims = toDimArray();
        int i;
        if (index >= 0) {
            i = (index);
        } else {
            i = (dimCount + index);
        }
        dims[i] = dimLength;
        return ProviderStore.shape(dims);
    }

    public long mulDims(int fromDim, int toDimExclusive) {
        int f;
        if (fromDim >= 0) {
            f = (fromDim);
        } else {
            f = (dimCount + fromDim);
        }
        int t;
        if (toDimExclusive >= 0) {
            t = (toDimExclusive);
        } else {
            t = (dimCount + toDimExclusive);
        }
        if (f < t) {
            int r = 1;
            for (int i = f; i < t; i++) {
                r *= at(i);
            }
            return r;
        }
        return 0;
    }

    public boolean isValid(int[] indices) {
        for (int i = indices.length - 1; i >= 0; i--) {
            int index = indices[i];
            if (index <= -1 || index >= at(i))
                return false;
        }
        return true;
    }

    public Shape removeDimAt(int axis) {
        int dimCount = this.dimCount;
        int[] result = new int[dimCount - 1];

        int _axis = (axis >= 0 ? axis : dimCount + axis);
        int index = 0;
        for (int i = 0; i < dimCount; i++) {
            if (_axis != i) {
                result[index++] = at(i);
            }
        }

        return ProviderStore.shape(result);
    }

    public Shape appendDim(int lastDimLength) {
        int dimCount = this.dimCount;
        int[] result = Arrays.copyOf(toDimArray(), dimCount + 1);
        result[dimCount] = lastDimLength;
        return ProviderStore.shape(result);
    }

    public int wrapNegIndex(int dimIndex) {
        if (dimIndex >= 0) {
            return (dimIndex);
        } else {
            return (dimCount + dimIndex);
        }
    }

    public double dataAt(double[] data, int[] indices, double ifOOB) {
        if (indices.length == 1 && indices[0] == 0 && data.length == 1)
            return data[0]; //scalar HACK

        if (isValid(indices))
            return data[calcDataIndex(indices)];
        else
            return ifOOB;
    }
}
