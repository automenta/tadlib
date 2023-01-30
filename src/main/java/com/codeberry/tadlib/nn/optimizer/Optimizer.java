package com.codeberry.tadlib.nn.optimizer;

import com.codeberry.tadlib.nn.optimizer.schedule.LearningRateSchedule;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;

public interface Optimizer {

    LearningRateSchedule learningRateSchedule();

    void optimize(List<Tensor> params);


}
