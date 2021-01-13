package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.ops.*;
import com.codeberry.tadlib.util.StringUtils;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.*;

import java.util.*;

import static com.codeberry.tadlib.memorymanagement.DisposalRegister.disposeAllExceptReturnedValues;
import static com.codeberry.tadlib.memorymanagement.DisposalRegister.registerForDisposal;
import static com.codeberry.tadlib.array.util.SoftmaxUtils.getSoftmaxGradientUpdates;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.*;
import static com.codeberry.tadlib.provider.opencl.OclDataType.*;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static com.codeberry.tadlib.provider.opencl.ops.Conv.ConvSize.Builder.convSizeBuilder;
import static java.lang.Math.*;
import static java.util.stream.Collectors.toList;


// for waiting: http://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.events.2pp.pdf
public class OclArray implements NDArray {
    public final OclBuffer buffer;
    private final InProgressResources resources;
    public final Shape shape;

    private OclArray(Shape shape, OclBuffer buffer, InProgressResources resources) {
        this.shape = shape;
        this.buffer = buffer.lockOrCreateLockedView();
        this.resources = resources;
    }

    private OclArray(Context context, double v) {
        this(ProviderStore.shape(), createBuffer(context, sizeOf(cl_double, 1), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        Memory nativeBuf = new Memory(cl_double.sizeOfElements(shape.getSize()));
        nativeBuf.setDouble(0, v);
        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                context.getQueue(), this.buffer, false, new SizeT(0), new SizeT(nativeBuf.size()),
                nativeBuf, null, resources.opEvent));
    }

    private OclArray(Context context, Shape shape, double v) {
        this(shape, createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        this.buffer.oclFill(context.getQueue(), resources, v);
    }

    private OclArray(Context context, Shape shape, double[] v) {
        this(shape, createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        Memory nativeBuf = new Memory(cl_double.sizeOfElements(shape.getSize()));
        nativeBuf.write(0, v, 0, v.length);
        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                context.getQueue(), this.buffer, false, new SizeT(0), new SizeT(nativeBuf.size()),
                nativeBuf, null, resources.opEvent));
    }

    public static OclArray createNDArray(Shape shape, OclBuffer buf, InProgressResources resources) {
        return registerForDisposal(new OclArray(shape, buf, resources));
    }

    public static OclArray createNDArray(Context context, double v) {
        return registerForDisposal(new OclArray(context, v));
    }

    public static OclArray createNDArray(Context context, Shape shape, double[] data) {
        return registerForDisposal(new OclArray(context, shape, data));
    }

    public static NDArray createNDArray(Context context, Shape shape, double v) {
        return registerForDisposal(new OclArray(context, shape, v));
    }

    public Pointer getArgPointer() {
        return buffer.argPointer;
    }

    @Override
    public void waitForValueReady() {
        Pointer ev = getKernelEvent();
        if (ev != null) {
            if (resources.isDisposed()) {
                throw new RuntimeException("Should not be disposed");
            }
            throwOnError(() -> OpenCL.INSTANCE.clWaitForEvents(1, OpenCL.PointerArray.pointers(ev)),
                    resources::getContentStatus);
        }
        resources.disposeDeep();
    }

    private static void tryReadBuffers(final String prefix, StringBuilder sb, List<OclBuffer> buffers) {
        for (int i = 0; i < buffers.size(); i++) {
            OclBuffer buf = buffers.get(i);

            sb.append(prefix).append("[").append(i).append("/").append(buffers.size()).append("]" +
                    "(").append(buf.getByteCount()).append("):")
                    .append(buf.canBeRead()).append("\n");
        }
    }

    private Pointer getKernelEvent() {
        if (this.resources != null && this.resources.opEvent != null) {
            return this.resources.opEvent.getValue();
        }
        return null;
    }

    @Override
    public NDArray negate() {
        return Simple.negate(buffer.context, this);
    }

    @Override
    public NDArray sqr() {
        return Simple.sqr(buffer.context, this);
    }

    @Override
    public NDArray sqrt() {
        return Simple.sqrt(buffer.context, this);
    }

    @Override
    public NDArray rot180(int yAxis, int xAxis) {
        return Simple.rot180(buffer.context, this, yAxis, xAxis);
    }

    @Override
    public NDArray pow(double val) {
        return Simple.pow(buffer.context, this, val);
    }

    @Override
    public NDArray add(NDArray other) {
        return Add.add(buffer.context, this, (OclArray) other);
    }

    @Override
    public NDArray add(double val) {
        return Add.add(buffer.context, this, val);
    }

    @Override
    public NDArray mul(NDArray other) {
        return Mul.mul(buffer.context, this, (OclArray) other);
    }

    @Override
    public NDArray div(NDArray other) {
        return Div.div(buffer.context, this, (OclArray) other);
    }

    @Override
    public NDArray mul(double val) {
        return Mul.mul(buffer.context, this, val);
    }

    @Override
    public NDArray div(double val) {
        return Div.div(buffer.context, this, val);
    }

    @Override
    public NDArray sum(Boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
        return Sum.sum(buffer.context, this, dimsToCollapse, keepRemove);
    }

    @Override
    public MaxPool2dResult maxPool2d(int size) {
        return MaxPool.maxPool2d(buffer.context, this, size);
    }

    @Override
    public NDArray maxPool2dGrad(MaxPool2dResult result) {
        return MaxPool.maxPool2dGrad(buffer.context, this, result);
    }

    @Override
    public ReluResult relu(double leakyScale) {
        return Relu.relu(buffer.context, this, leakyScale);
    }

    @Override
    public NDArray softmax() {
        return Softmax.softmax(buffer.context, this);
    }

    @Override
    public NDArray softMaxGrad(NDArray softmax, NDArray oneHotArray) {
        List<ValueUpdate> updates = getSoftmaxGradientUpdates(softmax, softmax.getShape().newIndexArray(),
                oneHotArray, 0);

        return softmax.withUpdates(updates).mul(this);
    }

    /**
     * Reads data from OpenCL, then modify it and create result array.
     * <p>
     * TODO: Improve. Maybe create a mask that is multiplied with this array, to avoid reading from OpenCL.
     */
    @Override
    public DropOutResult dropOut(Random rnd, double dropoutKeep) {
        double[] vals = readFlatArray();

        double gradValue = 1.0 / dropoutKeep;
        double[] gradMaskData = new double[vals.length];
        for (int i = 0; i < vals.length; i++) {
            if (rnd.nextDouble() >= dropoutKeep) {
                vals[i] = 0;
            } else {
                vals[i] /= dropoutKeep;
                gradMaskData[i] = gradValue;
            }
        }

        NDArray output = ProviderStore.array(vals, shape);
        return new DropOutResult() {
            @Override
            public NDArray getOutput() {
                return output;
            }

            @Override
            public NDArray createMask() {
                return ProviderStore.array(gradMaskData, shape);
            }
        };
    }

    private double[] readFlatArray() {
        int size = toIntExact(shape.getSize());
        Memory nativeBuf = readToNative();
        double[] vals = new double[size];
        nativeBuf.read(0, vals, 0, vals.length);
        return vals;
    }

    @Override
    public NDArray withUpdates(List<ValueUpdate> updates) {
        return Update.update(buffer.context, this, updates);
    }

    @Override
    public NDArray conv2d(NDArray filter, int offsetY, int offsetX) {
        int filterHeight = filter.getShape().at(0);
        int filterWidth = filter.getShape().at(1);

        int outHeight = shape.at(-3);
        int outWidth = shape.at(-2);

        int diffY = getShape().at(-3) - outHeight;
        int diffX = getShape().at(-2) - outWidth;

        return conv2d(filter,
                offsetY - (filterHeight - 1) / 2 + diffY / 2,
                offsetX - (filterWidth - 1) / 2 + diffX / 2,
                outHeight, outWidth);
    }

    @Override
    public NDArray conv2d(NDArray filter, int offsetY, int offsetX, int outHeight, int outWidth) {
        return Conv.conv2d(buffer.context, this, (OclArray) filter,
                new int[]{
                        offsetY,
                        offsetX},
                convSizeBuilder()
                        .height(outHeight)
                        .width(outWidth)
                        .build());
    }

//    @Override
//    public NDArray calcConv2dFilterGradient(NDArray input, NDArray filter) {
////        System.out.println("============");
//        Shape filterShape = filter.getShape();
////        System.out.println("Filter.shape: "+ filterShape);
//        testNan(this);
//        NDArray gradTransposed = this.transpose(1, 2, 0, 3);
//        testNan(gradTransposed);
////        print("== Grad", this, gradTransposed, 3,3);
//
//        testNan(input);
//        NDArray inputTransposed = input.transpose(3, 1, 2, 0);
//        testNan(inputTransposed);
////        print("== Input", input, inputTransposed, 3,3);
//
//        int offsetY = -(filter.getShape().at(0)-1)/2;
//        int offsetX = -(filter.getShape().at(1)-1)/2;
//        NDArray tmpOut = ((OclArray)inputTransposed).conv2dTry(gradTransposed, offsetY, offsetX,
//                filterShape.at(0),
//                filterShape.at(1));
//        testNan(tmpOut);
////        System.out.println("tmpOut.shape: "+tmpOut.getShape());
//
//        NDArray out = tmpOut.transpose(1, 2, 0, 3);
//        testNan(out);
//        //NDArray outRot180 = out.rot180(0, 1);
////        System.out.println("out.shape: "+out.getShape());
////        System.out.println("out: " + toJson(out.toDoubles(), COMPACT));
//
//        //return outRot180;
//        return out;
//        //Inp: (batch, Hi, Wi, Ci)
//        //
//        //Flt: (Hf, Wf, Ci, Co)
//        //
//        //Out: (batch, Ho, Wo, Co)
//        //
//        //
//        //Grd transpose: (Co, Ho, Wo, batch)
//        //
//        //Inp transpose: (Hi, Wi, batch, Ci)
//        //
//        //Res:           (Co, Hf, Wf, Ci) -> (Hf, Wf, Ci, Co)
//    }

    @Override
    public NDArray calcConv2dFilterGradient(NDArray input, NDArray filter) {
        return disposeAllExceptReturnedValues(() -> {
            Shape filterShape = filter.getShape();
            NDArray gradTransposed = this.transpose(1, 2, 0, 3);
            NDArray inputTransposed = input.transpose(3, 1, 2, 0);

            int offsetY = -(filter.getShape().at(0) - 1) / 2;
            int offsetX = -(filter.getShape().at(1) - 1) / 2;
            NDArray tmpOut = inputTransposed.conv2d(gradTransposed, offsetY, offsetX,
                    filterShape.at(0),
                    filterShape.at(1));

            return tmpOut.transpose(1, 2, 0, 3);
        });
    }

    public void dispose() {
        this.buffer.dispose();
        this.resources.disposeDeep();
    }

    @Override
    public NDArray matmul(NDArray b) {
        return MatMul.matmul(buffer.context, this, (OclArray) b);
    }

    @Override
    public NDArray transpose(int... axes) {
        return Transpose.transpose(buffer.context, this, axes);
    }

    @Override
    public Shape getShape() {
        return shape;
    }

    @Override
    public NDArray reshape(int... dims) {
        return createNDArray(this.shape.reshape(dims), buffer, resources);
    }

    @Override
    public Object toDoubles() {
        Memory nativeBuf = readToNative();

        return FlatToMultiDimArrayConverter.toDoubles(this.shape,
                i -> nativeBuf.getDouble(i * cl_double.byteSize));
    }

    private Memory readToNative() {
        Memory nativeBuf = new Memory(cl_double.sizeOfElements(shape.getSize()));

        Pointer kernelEvent = getKernelEvent();

        OpenCL.PointerArray events = (kernelEvent != null ? OpenCL.PointerArray.pointers(kernelEvent) : null);

        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueReadBuffer(buffer.context.getQueue(),
                buffer, true,
                new SizeT(0),
                new SizeT(nativeBuf.size()), nativeBuf, events,
                null));

        resources.disposeDeep();

        return nativeBuf;
    }

    @Override
    public NDArray subArray(int fromBatchIndex, int fromOffset, int endBatchIndex, int toOffset) {

        InProgressResources res = new InProgressResources(buffer.context);
        OclBuffer buf = this.buffer.oclDoubleSubBuffer(res, fromOffset, toOffset);
        int[] dims = this.shape.toDimArray();
        dims[0] = endBatchIndex - fromBatchIndex;

        return createNDArray(ProviderStore.shape(dims), buf, res);
    }

    @Override
    public NDArray normalOrderedCopy() {
        return this;
    }

    @Override
    public double[] getInternalData() {
        return readFlatArray();
    }

    @Override
    public double dataAt(int... indices) {
        int index = getShape().calcDataIndex(indices);
        return buffer.oclReadDouble(resources, index);
    }

    /**
     * Meant for tracking & debugging buffers that was unintentionally released.
     */
    public void setBufferDisposeCallback(Runnable r) {
        this.buffer.setDisposeCallback(r);
    }

    @Override
    public String toString() {
        return StringUtils.toString(this);
    }


    /**
     * Holds on to objects until disposed/finalized.
     * <p>
     * For preventing GC of source objects.
     */
    public static class InProgressResources {
        private final Context context;

        public volatile OclEventByReference opEvent;
        private List<Memory> memoryPointers = new ArrayList<>();
        private List<ByReference> references = new ArrayList<>();
        private List<OclEventByReference> events = new ArrayList<>();
        private List<OclBuffer> disposableBuffers = new ArrayList<>();
        private List<OclBuffer> strongReferredBuffers = new ArrayList<>();
        private List<OclArray> strongReferredNDArrays = new ArrayList<>();

        public InProgressResources(Context context) {
            this.context = context;

            opEvent = new OclEventByReference();
            events.add(opEvent);
        }

        public void registerDependency(OclArray dependantArray) {
            this.strongReferredNDArrays.add(dependantArray);
        }

        public void registerDisposableBuffer(OclBuffer dependantArray) {
//            (new Exception("Registered mem: " + Pointer.nativeValue(dependantArray))).printStackTrace();
            this.disposableBuffers.add(dependantArray);
        }

        public void registerReferredBuffer(OclBuffer keepRefToBuffer) {
//            (new Exception("Keep mem: " + Pointer.nativeValue(keepRefToBuffer))).printStackTrace();
            this.strongReferredBuffers.add(keepRefToBuffer);
        }

        public OpenCL.PointerArray getDependencyEvents() {
            Pointer[] events = toEventPointers(strongReferredNDArrays);
            if (events != null) {
                return OpenCL.PointerArray.pointers(events);
            }
            return null;
        }

        private static Pointer[] toEventPointers(List<OclArray> arrList) {
            int count = countEvents(arrList);
            if (count > 0) {
                return writeKernelEvents(arrList, count);
            }
            return null;
        }

        private static Pointer[] writeKernelEvents(List<OclArray> arrList, int count) {
            Pointer[] events = new Pointer[count];

            int evIdx = 0;
            for (OclArray arr : arrList) {
                Pointer ev = arr.getKernelEvent();
                if (ev != null) {
                    events[evIdx++] = ev;
                }
            }
            if (evIdx == events.length) {
                return events;
            }
            System.err.println("Should not happen");
            return Arrays.copyOf(events, evIdx);
        }

        private static int countEvents(List<OclArray> arrList) {
            int count = 0;
            for (OclArray arr : arrList) {
                if (arr.getKernelEvent() != null) {
                    count++;
                }
            }
            return count;
        }

        public Pointer argLong(long v) {
            LongByReference ref = new LongByReference(v);
            references.add(ref);

            return ref.getPointer();
        }

        public Pointer argInt(int v) {
            IntByReference ref = new IntByReference(v);
            references.add(ref);

            return ref.getPointer();
        }

        public Pointer argDouble(double v) {
            DoubleByReference ref = new DoubleByReference(v);
            references.add(ref);

            return ref.getPointer();
        }

        public synchronized boolean isDisposed() {
            return opEvent == null;
        }

        public synchronized void disposeDeep() {
            for (OclArray ndArr : strongReferredNDArrays) {
                ndArr.resources.disposeDeep();
            }
            for (OclEventByReference event : events) {
                event.oclRelease();
            }
            for (OclBuffer buf : disposableBuffers) {
                buf.dispose();
            }

            strongReferredNDArrays.clear();
            strongReferredBuffers.clear();
            disposableBuffers.clear();
            memoryPointers.clear();
            references.clear();
            events.clear();
            opEvent = null;
        }

        public Pointer registerReadOnlyArg(int[] v) {
            if (v != null) {
                Memory memory = createMemory(v);
                OclBuffer buf = createBuffer(context, sizeOf(cl_int, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
                OclEventByReference bufferWriteEvt = new OclEventByReference();
                throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                        context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                        memory, null, bufferWriteEvt));

                Memory pointerToPointer = new Memory(Native.POINTER_SIZE);
                pointerToPointer.setPointer(0, buf);

                memoryPointers.add(memory);
                memoryPointers.add(pointerToPointer);
                events.add(bufferWriteEvt);

                registerDisposableBuffer(buf);

                return pointerToPointer;
            }
            return Pointer.NULL;
        }

        public Pointer registerReadOnlyArg(long[] v) {
            if (v != null) {
                Memory memory = createMemory(v);
                OclBuffer buf = createBuffer(context, sizeOf(cl_long, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
                OclEventByReference bufferWriteEvt = new OclEventByReference();
                throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                        context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                        memory, null, bufferWriteEvt));

                Memory pointerToPointer = new Memory(Native.POINTER_SIZE);
                pointerToPointer.setPointer(0, buf);

                memoryPointers.add(memory);
                memoryPointers.add(pointerToPointer);
                events.add(bufferWriteEvt);

                registerDisposableBuffer(buf);

                return pointerToPointer;
            }
            return Pointer.NULL;
        }

        public Pointer registerReadOnlyArg(double[] v) {
            if (v != null) {
                Memory memory = createMemory(v);
                OclBuffer buf = createBuffer(context, sizeOf(cl_double, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
                OclEventByReference bufferWriteEvt = new OclEventByReference();
                throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                        context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                        memory, null, bufferWriteEvt));

                Memory pointerToPointer = new Memory(Native.POINTER_SIZE);
                pointerToPointer.setPointer(0, buf);

                memoryPointers.add(memory);
                memoryPointers.add(pointerToPointer);
                events.add(bufferWriteEvt);

                registerDisposableBuffer(buf);

                return pointerToPointer;
            }
            return Pointer.NULL;
        }

        private static Memory createMemory(int[] intArr) {
            Memory dimBlockSizesLeftPointer = new Memory(cl_int.sizeOfElements(intArr.length));
            dimBlockSizesLeftPointer.write(0, intArr, 0, intArr.length);
            return dimBlockSizesLeftPointer;
        }

        private static Memory createMemory(long[] longArr) {
            Memory dimBlockSizesLeftPointer = new Memory(cl_long.sizeOfElements(longArr.length));
            dimBlockSizesLeftPointer.write(0, longArr, 0, longArr.length);
            return dimBlockSizesLeftPointer;
        }

        private static Memory createMemory(double[] doubleArr) {
            Memory dimBlockSizesLeftPointer = new Memory(cl_double.sizeOfElements(doubleArr.length));
            dimBlockSizesLeftPointer.write(0, doubleArr, 0, doubleArr.length);
            return dimBlockSizesLeftPointer;
        }

        public void registerDependencyEvent(OclEventByReference event) {
            Objects.requireNonNull(event, "Cannot add NULL dependency event");

            events.add(event);
        }

        private String getContentStatus() {
            StringBuilder sb = new StringBuilder();
            tryReadBuffers("StrongRefBuf", sb, strongReferredBuffers);
            tryReadBuffers("DisposableRefBuf", sb, disposableBuffers);
            tryReadBuffers("NDArray.buffer", sb, extractBuffers(strongReferredNDArrays));
            for (int i = 0; i < memoryPointers.size(); i++) {
                Memory m = memoryPointers.get(i);
                sb.append("Memory[").append(i).append("/").append(memoryPointers.size()).append("]:").append(m.size()).append(":readOk=");
                try {
                    byte[] tmp = new byte[toIntExact(m.size())];
                    m.read(0, tmp, 0, tmp.length);
                    sb.append("true");
                } catch (Exception e) {
                    sb.append("false").append(e.toString());
                }
                sb.append("\n");
            }
            for (int i = 0; i < references.size(); i++) {
                ByReference ref = references.get(i);
                sb.append("Reference[").append(i).append("/").append(references.size()).append("]:").append(ref).append("\n");
            }

            OpenCL.PointerArray events = getDependencyEvents();
            if (events != null) {
                for (int i = 0; i < events.length(); i++) {
                    sb.append("Event[").append(i).append("/").append(events.length()).append("]:waitRet=")
                            .append(OpenCL.INSTANCE.clWaitForEvents(1, OpenCL.PointerArray.pointers(events.getPointer((long) i * Native.POINTER_SIZE))))
                            .append("\n");
                }
            }

            return sb.toString();
        }

        private static List<OclBuffer> extractBuffers(List<OclArray> arrays) {
            return arrays.stream()
                    .map(arr -> arr.buffer)
                    .collect(toList());
        }
    }
}