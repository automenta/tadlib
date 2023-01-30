package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.nn.Initializer;
import com.codeberry.tadlib.nn.optimizer.SGD;
import com.codeberry.tadlib.nn.optimizer.schedule.FixedLearningRate;
import com.codeberry.tadlib.provider.java.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.random.RandomGenerator;
import java.util.stream.DoubleStream;

import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.TensorTrainSimple.random;

/**
 * REINFORCE - Policy Gradient
 *
 * TODO not working yet
 *
 * translation from: https://gist.github.com/SeunghyunSEO/ce6732edb4adf4f4c41821b4cd6d3337
 * Karpathy's original:
 *   http://karpathy.github.io/2016/05/31/rl/
 *   https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
 *   https://www.youtube.com/watch?v=tqrcjHuNdmQ
 *
 * https://www.datahubbs.com/reinforce-with-pytorch/
 * https://github.com/ynuwm/pytorch-reinforce-algorithm/blob/master/reinforce_continuous.py
 * https://github.com/indigoLovee/Reinforce_pytorch/blob/main/Reinforce_continuous.py
 *
 * https://www.reddit.com/r/reinforcementlearning/comments/cahzjj/comment/et97plk/?utm_source=share&utm_medium=web2x&context=3
 * Think about it this way. Policy Gradients increase the probability of good trajectories, and decline the probability of bad trajectories. Good and bad is measured by Reward. If the reward is negative, the trajectory is bad, if it is positive it is good. All action probabilities along the trajectory are manipulated equally only depending on the reward you are getting along the way.
 * If the action was good, the mean of the Gaussian should go in the direction of higher probability (i.e. higher mean), and the variance should be decreased. If the action was bad, the mean of the Gaussian should get lower but with higher variance. For the actual policy you use a Neural Network (or other Function) which as a last "layer/activation function" gets fed into the PDF of the desired Gaussian. Now your network puts out a density, making it a policy(!). So lets apply PG to it.
 * The vanilla Policy Gradient is E[R \grad \log \pi_\theta]. That is literally your gradient. So you perform a log on your policy output (the PDF of the Gaussian) along the whole trajectory, build the gradient with respect to the input. Multiply that by R (thus reversing direction for bad trajectories), and build the average over all your samples.
 * If you want to use Pytorch or Tensorflow, both have numerically stable implementations for the \grad \log \pi step. You definitly should use those.
 */
public class Reinforce {
    public static void main(String[] args) {
        final Random rng = new Random(0);

        int i = 2;
        int a = 2;
        int h = 16;
        Reinforce r = new Reinforce(i, h, a);
        r.clear(rng);
        double[] x = random(rng, i);
        for (int e = 0; e < 10000; e++) {
            double reward = 0.5f * (1/(1+Math.abs(x[0]-0.5)) + 1/(1 + Math.abs(x[1] - 0.5)));
            double[] act = r.learn(null, null, reward, x);
            System.out.println(Arrays.toString(x) + " -> " + reward + " => " + Arrays.toString(act));
            x[0] = Math.min(1, Math.max(0, x[0] + (act[0]-0.5) * 0.1));
            x[1] = Math.min(1, Math.max(0, x[1] + (act[1]-0.5) * 0.1));
        }
    }
    /*
    /*
class ReinforceAgent:
    """
    ReinforceAgent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma):
        # set values such that initial conditions correspond to left-epsilon greedy
        self.theta = np.array([-1.47, 1.47])
        self.alpha = alpha
        self.gamma = gamma
        # first column - left, second - right
        self.x = np.array([[0, 1],
                           [1, 0]])
        self.rewards = []
        self.actions = []

    def get_pi(self):
        h = np.dot(self.theta, self.x)
        t = np.exp(h - np.max(h))
        pmf = t / np.sum(t)
        # never become deterministic,
        # guarantees episode finish
        imin = np.argmin(pmf)
        epsilon = 0.05

        if pmf[imin] < epsilon:
            pmf[:] = 1 - epsilon
            pmf[imin] = epsilon

        return pmf

    def choose_action(self, reward):
        if reward is not None:
            self.rewards.append(reward)

        pmf = self.get_pi()
        go_right = np.random.uniform() <= pmf[1]
        self.actions.append(go_right)

        return go_right

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * G[i] * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []
     */

    @Deprecated
    final int D, H, O; // number of hidden layer neurons
    private final Tensor I;
    private final Tensor model;
//    private final Tensor modelTarget;
    private final Tensor actionSample;
    private final Tensor dlogp;

    //TODO make this vary
    int episodeLen = 2; // every how many episodes to do a param update?

    public final float learn =
        //1.0e-4f
        //3e-4f
        1.0e-2f
        //5.0e-2f
    ;

    public final float gamma =
        0.1f
        //0.5f
        //0.9f
        //0.99f;// # discount factor for reward
    ;

    //float decay_rate = 0.99;// # decay factor for RMSProp leaky sum of grad^2

    final List<Double> R = new ArrayList<>(episodeLen);

    int iteration = 0, episode_number = 0;

    final RandomGenerator rng = new Random(0);
    final SGD opt;

    public Reinforce(int inputs, int hidden, int actions) {
        this.D = inputs;
        this.H = hidden;
        this.O = actions;
        this.I = new Tensor(Shape.shape(1, inputs));
        this.model = DENSE(  RELU(DENSE(I, H)), actions/* * 2 */);
//        this.modelTarget = new Tensor(new double[actions]);
        this.actionSample = new Tensor(new double[actions]);
        this.dlogp = log_prob(actionSample, model);
        //this.loss = SUM(SQR(SUB(model, modelTarget)));
        this.opt = new SGD(new FixedLearningRate(learn));
    }

    private static double[] discountRewards(List<Double> rewards, double gamma) {
        int n = rewards.size();
        double[] d = new double[n];
        double runningSum = 0;
        for (int t = n - 1; t >= 0; t--) {
            runningSum = runningSum * gamma + rewards.get(t); //TODO fma
            d[t] = runningSum;
        }

        /*
            standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
         */
        double[] mv = variance(DoubleStream.of(d));
        double mean = mv[0], std = Math.sqrt(mv[1]);
        if (std > Double.MIN_NORMAL) {
            for (int i = 0; i < d.length; i++)
                d[i] = (d[i] - mean) / std;
        }

        return d;
    }

    @Deprecated private static double[] variance(DoubleStream s) {
        List<Double> dd = new ArrayList<>();
        double[] mean = {0};
        s.forEach(e -> {
            dd.add(e);
            mean[0] += e;
        });
        if (dd.isEmpty())
            return null;
        mean[0] /= dd.size();

        double variance = 0;
        int n = dd.size();
        for (double aDouble : dd) {
            double d = (aDouble - mean[0]);
            variance += d * d;
        }

        variance /= n;

        return new double[]{mean[0], variance};
    }
    /*


        def policy_forward(x):
          h = np.dot(model['W1'], x)
          h[h<0] = 0 # ReLU nonlinearity
          logp = np.dot(model['W2'], h)
          p = sigmoid(logp)
          return p, h # return probability of taking action 2, and hidden state

        def policy_backward(eph, epdlogp):
          """ backward pass. (eph is array of intermediate hidden states) """
          dW2 = np.dot(eph.T, epdlogp).ravel()
          dh = np.outer(epdlogp, model['W2'])
          dh[eph <= 0] = 0 # backpro prelu
          dW1 = np.dot(dh.T, epx)
          return {'W1':dW1, 'W2':dW2}

     */
    public void clear(Random rng) {
        model.init(new Initializer.UniformInitializer(rng, 0.5f));
    }

    final List<double[]> X = new ArrayList<>(), DLOGP = new ArrayList<>();


    public double[] learn(double[] xPrev, double[] actionPrev, double reward, double[] x) {
//          # preprocess the observation, set input to network to be difference image
//          cur_x = prepro(observation)
//          x = cur_x - prev_x if prev_x is not None else np.zeros(D)
//          prev_x = cur_x

        //# forward the policy network and sample an action from the returned probability
//        double[] h = model.layers[0].forward(x, rng);
//        double[] aprob = model.layers[1].forward(h, rng);
        this.I.set(x);
        double[] aprob = model.val().data;



//        boolean discreteActions = false;
        double[] actionSample;
//        if (discreteActions) {
//            int theAction = decide.applyAsInt(aprob);
//            action = new double[O];
//            action[theAction] = 1;
//
//            dlogp = new double[O];
//            for (int i = 0; i < O; i++) {
//                dlogp[i] = aprob[i] - action[i];  //grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
//                //dlogp[i] = action[i] - aprob[i];  //grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
//                //dlogp[i] =Math.log(1+aprob[i]) - action[i];
//                //dlogp[i] = aprob[i];
//                //dlogp[i] = action[i];
//                //dlogp[i] = Math.log(aprob[i]) - action[i];
//                //dlogp[i] = Math.log(aprob[i]) - Math.log(action[i]);
//                //dlogp[i] = Math.log(aprob[i]);
//            }
//            this.grad = dlogp;
//        } else {
            @Deprecated double SIGMA = 0.25f; @Deprecated double[] asigma = new double[aprob.length]; Arrays.fill(asigma, SIGMA);
            actionSample = normalSample(aprob, asigma);
            this.actionSample.set(actionSample);
            //dlogp = log_prob(actionSample, aprob, asigma);
            //for (int i = 0; i < dlogp.length; i++) dlogp[i] -= action[i];
            //dlogp = new double[aprob.length]; for (int i = 0; i < dlogp.length; i++) dlogp[i] = aprob[i] - action[i];
//        }



        X.add(x.clone()); //observation
//        hs.add(h.clone()); //hidden state
        DLOGP.add((double[]) this.dlogp.val().toDoubles());
        R.add(reward);

        if (++iteration % episodeLen == 0)
            episode();

        return actionSample;
    }

    private void episode() {
        var edlogp = DLOGP;

        //compute the discounted reward backwards through time
        double[] discountER = discountRewards(R, gamma);

        int E = edlogp.size();

        //# modulate the gradient with advantage (PG magic happens right here.)
        for (int j = 0; j < E; j++) {
            double[] ej = edlogp.get(j);

            double[] pg = new double[O];
            double dj = discountER[j];

            for (int i = 0; i < ej.length; i++)
                pg[i] += ej[i] * dj;
                //pg[i] -= ej[i] * dj;

            I.set(X.get(j));
            //model.backward(pg, true);
            model.optimize(opt, pg);
            //modelTarget.set(pg);
            //loss.optimize(opt);
        }
//        Util.mul(1f / E, pg); //mean?
//        this.grad = pg; //HACK



//        for (int i = 0; i < E; i++) {
//            //dW2 = np.dot(eph.T, epdlogp).ravel() ...
//            double[] ephi = eph.get(i);
//            double[] epdlogi = epdlogp.get(i);
//            //Util.mul(epdlogi, ephi);
//            double[] dw = mult(ephi, epdlogi, ephi.length, epdlogi.length, 1);
//            //Util.mul(-1, dw);
//
//            AbstractLayer[] l = model.layers;
//            for (AbstractLayer x : l)
//                x.startNext();
//
//            model.updater.reset(model.weightCount(), 1);
//            LinearLayer lastLayer = (LinearLayer) l[l.length - 1];
//            //((BatchWeightUpdater) model.updater).commitGrad(lastLayer, dw);
//            for (int j = 0; j < dw.length; j++)
//                lastLayer.W[j] += dw[j];


//        }

//        """ backward pass. (eph is array of intermediate hidden states) """

        //double[] grad = model.putDelta();
        /*

        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
          for k,v in model.items():
            g = grad_buffer[k] # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer


        */

        X.clear();
        DLOGP.clear();
        R.clear(); //# reset episode memory
        episode_number++;
    }


    static final private double c = Math.log(Math.sqrt(2 * Math.PI));

    /** https://github.com/pytorch/pytorch/blob/a27a4a02fecfdd626b25794a84954731b80f29fb/torch/distributions/normal.py#L77 */
    public static Tensor log_prob(Tensor x, Tensor mean) {
        @Deprecated double asigma = 0.25f;
        double sigmaI = asigma;//sigma[i];
        double varI = (sigmaI*sigmaI);
        double logSigmaI = Math.log(sigmaI);
        return ADD(MUL(NEGATE(SQR(SUB(x, mean))), 1/(2*varI)), -logSigmaI - c);
    }

    public double[] normalSample(double[] mean, double[] sigma) {
        double[] x = new double[mean.length];
        for (int i = 0; i < mean.length; i++) {
            double xi = rng.nextGaussian(mean[i], sigma[i]);
            xi = Math.max(0, Math.min(1, xi));
            x[i] = xi;
        }
        return x;
    }
}
