import numpy as np
import math


# d_model: the parameter of the model
# N: length of the input Sequence
# M:  HBM, on-chip SRAM size


class FlashAttention():

    def __init__(self, d_model: int, N: int, M: int):
        # d_model = 128 and N = 512 default
        self.d_model = d_model
        self.N = N
        # default M = 1028
        self.M = M

        assert (M % d_model == 0), "M not divisible by d_model"

        self.Bc = (M//(4*d_model))
        self.Br = min(d_model, self.Bc)

        assert (N % self.Br == 0 or N %
                self.Bc == 0), "Sequence Length N could not be divided into the smaller SRAM chunks"

        self.Tr = N//self.Br
        self.Tc = N//self.Bc
        # print(self.Bc)
        # print(self.Br)
        # print(self.Tc)
        # print(self.Tr)

    def max_of_each_row(self, Sij):
        m_temp = np.zeros((len(Sij), 1))

        return m_temp

    def computeContextAwareOutput(self):

        Output = np.zeros((self.N, self.d_model))

        l = np.zeros((self.N))
        m = np.full((self.N), -np.inf)

        # generating the random Q, V and K matrices
        # initialising random variables between -10 and 10

        Q = np.random.rand(self.N, self.d_model) * 20 - 10
        K = np.random.rand(self.N, self.d_model) * 20 - 10
        V = np.random.rand(self.N, self.d_model) * 20 - 10

        for j in range(0, self.Tc):
            K_temp = K[j:j+self.Bc, :]
            V_temp = V[j:j+self.Bc, :]

            for i in range(0, self.Tr):
                Q_temp = Q[i:i+self.Br, :]
                O_temp = Output[i:i+self.Br, :]
                l_temp = l[i:i+self.Br]
                m_temp = m[i:i+self.Br]
                mij = np.zeros((self.Br))
                pij = np.zeros((self.Br, self.Bc))
                lij = np.zeros((self.Br))
                mi_new = np.zeros((self.Br))
                li_new = np.zeros((self.Br))

                # computing Sij
                Sij = np.einsum('ij,jk->ik', Q_temp, K_temp.T)

                # computing mi
                for k in range(0, self.Br):
                    curr_max = -math.inf
                    for index in range(0, self.Bc):
                        curr_max = max(curr_max, Sij[k][index])
                    mij[k] = curr_max

                # computing Pij
                for row_index in range(0, self.Br):
                    for col_index in range(0, self.Bc):
                        pij[row_index][col_index] = np.exp(
                            Sij[row_index][col_index] - mij[row_index])

                # computing lij
                for k in range(0, self.Br):
                    row_sum = 0
                    for index in range(0, self.Br):
                        row_sum = row_sum+pij[k][index]
                    lij[k] = row_sum

                # computing mi_new and li_new
                for k in range(0, self.Br):
                    mi_new[k] = max(m_temp[i], mij[k])
                    li_new[k] = (np.exp(m[k]-mi_new[k])*l_temp[k]) + \
                        (np.exp(mij[k]-mi_new[k]) * lij[k])

                li_diag = np.diag(li_new)

                # updating output

                for k in range(i, i+self.Br):
                    Output[k] =

                # updating m and l

                m[i:i+self.Br] = mi_new[i:i+self.Br]
                l[i:i+self.Br] = li_new[i:i+self.Br]


instance = FlashAttention(4, 32, 16)
instance.computeContextAwareOutput()
