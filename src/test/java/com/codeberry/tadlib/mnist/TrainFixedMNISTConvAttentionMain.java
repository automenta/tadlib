package com.codeberry.tadlib.mnist;

import com.codeberry.tadlib.nn.optimizer.RMSProp;

import static com.codeberry.tadlib.mnist.SimpleTrainer.TrainParams.trainParams;
import static com.codeberry.tadlib.nn.optimizer.schedule.DecayingLearningRate.decayingLearningRate;
import static com.codeberry.tadlib.nn.optimizer.schedule.SawToothSchedule.sawTooth;

public class TrainFixedMNISTConvAttentionMain {

    public static void main(String[] args) {

        SimpleTrainer trainer = new SimpleTrainer(trainParams("Att")
                .batchSize(32)
                // Regular MNIST
                .optimizer(new RMSProp(
                        sawTooth(decayingLearningRate(0.0003, 0.0000001, 10, 0.8),
                                0.3, 6)))
//                .optimizer(new SGD(0.15))
                .loaderParams(MNISTLoader.LoadParams.params()
                        .loadFashionMNIST()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(FixedMNISTConvAttentionModel.Factory.Builder.factoryBuilder()
                        .firstConvChannels(16)
                        .secondConvChannels(40)
                        .l2Lambda(0.01)
                        .weightInitRandomSeed(4)
                        .attentionLayers(2)
                        .multiAttentionHeads(8)
                        .dropoutKeep(0.75)
                        .build()));

        trainer.trainEpochs(50000);
    }
}
