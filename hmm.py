import numpy as np


class HMM:
    """
    Order 1 Hidden Markov Model
    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    Common Variables
    ----------------
    obs_seq : list of int
        list of observations (represented as ints corresponding to output
        indexes in B) in order of appearance
    T : int
        number of observations in an observation sequence
    N : int
        number of states
    """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N, T))
        F[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N, T))
        X[:, -1:] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[n, t] = np.sum(X[:, t + 1] * self.A[n, :] * self.B[:, obs_seq[t + 1]])

        return X

    def observation_prob(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self._forward(obs_seq)[:, -1])

    def simulate(self, T):

        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t - 1], :])
            observations[t] = draw_from(self.B[states[t], :])
        return observations

    def baum_welch_train(self, observations, criterion=0.05):
        n_states = self.A.shape[0]
        n_samples = len(observations)

        done = False
        while not done:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = self._forward(observations)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self._backward(observations)

            xi = np.zeros((n_states, n_states, n_samples - 1))
            for t in range(n_samples - 1):
                denom = np.dot(np.dot(alpha[:, t].T, self.A) * self.B[:, observations[t + 1]].T, beta[:, t + 1])
                for i in range(n_states):
                    numer = alpha[i, t] * self.A[i, :] * self.B[:, observations[t + 1]].T * beta[:, t + 1].T
                    xi[i, :, t] = numer / denom

            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.squeeze(np.sum(xi, axis=1))
            # Need final gamma element for new B
            prod = (alpha[:, n_samples - 1] * beta[:, n_samples - 1]).reshape((-1, 1))
            gamma = np.hstack((gamma, prod / np.sum(prod)))  # append one more to gamma!!!

            newpi = gamma[:, 0]
            newA = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
            newB = np.copy(self.B)

            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma, axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:, lev] = np.sum(gamma[:, mask], axis=1) / sumgamma

            if np.max(abs(self.pi - newpi)) < criterion and \
                    np.max(abs(self.A - newA)) < criterion and \
                    np.max(abs(self.B - newB)) < criterion:
                done = 1

            self.A[:], self.B[:], self.pi[:] = newA, newB, newpi
            return self.A, self.B, self.pi
