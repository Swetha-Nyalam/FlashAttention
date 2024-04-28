import numpy as np
import torch
import math


class Attention:
    def __init__(self, Q, V, K):
        self.Q = Q
        self.V = V
        self.K = K
        self.N = Q.shape[1]
        self.d_model = Q.shape[0]

    def softmax(self, S):
        S_norm = np.exp(S-np.max(S, axis=0, keepdims=True))
        return np.divide(S_norm, np.sum(S_norm, axis=0, keepdims=True))

    def calculate_attention(self):
        S = (np.matmul(Q, K.T))/(math.sqrt(self.d_model))
        P = self.softmax(S)
        O = np.matmul(P, self.V)
        return O


class FlashAttention:
    def __init__(self, Q, K, V, M):
        self.Q = Q
        self.K = K
        self.V = V
        self.M = M
        self.N = Q.shape[0]
        self.d_model = Q.shape[1]
        # Set block sizes 𝐵𝑐 =  (𝑀/4𝑑) , 𝐵𝑟 = min ((𝑀/4𝑑) , d)
        self.Bc = math.ceil(self.M/(4*d_model))
        self.Br = min(self.Bc, self.d_model)
        self.O = np.zeros((self.N, self.d_model))
        self.l = np.zeros((self.N))
        self.m = np.full(self.N, -np.inf)

    def compute_flash_attention(self):
        for j in range(0, self.N, self.Bc):
            # Load K𝑗 , V𝑗 from HBM to on-chip SRAM
            K_j = self.K[j:j+self.Bc, :]
            V_j = self.V[j:j+self.Bc, :]

            for i in range(0, self.N, self.Br):
                # Load Q𝑖 , O𝑖 , ℓ𝑖 , 𝑚𝑖 from HBM to on-chip SRAM
                Q_i = self.Q[i:i+self.Br, :]
                O_i = self.O[i:i+self.Br, :]
                l_i = self.l[i:i+self.Br]
                m_i = self.m[i:i+self.Br]
                # On chip, compute S𝑖 𝑗 = Q𝑖K𝑇 𝑗 ∈ R 𝐵𝑟×𝐵𝑐
                S_ij = np.matmul(Q_i, K_j.T)
                # On chip, compute 𝑚˜𝑖 𝑗 = rowmax(S𝑖 𝑗) ∈ R 𝐵𝑟 , P˜ 𝑖 𝑗 = exp(S𝑖 𝑗 − 𝑚˜𝑖 𝑗) ∈ R 𝐵𝑟×𝐵𝑐 (pointwise), ℓ˜ 𝑖 𝑗 = rowsum(P˜ 𝑖 𝑗) ∈ R 𝐵𝑟
                m_ij = np.max(S_ij, axis=1, keepdims=False)
                P_ij = np.exp(S_ij-m_ij)
                l_ij = np.sum(P_ij, axis=0, keepdims=True)
                # On chip, compute 𝑚 new 𝑖 = max(𝑚𝑖 , 𝑚˜𝑖 𝑗) ∈ R 𝐵𝑟 , ℓ new 𝑖 = 𝑒 𝑚𝑖−𝑚new 𝑖 ℓ𝑖 + 𝑒 𝑚˜𝑖 𝑗−𝑚new 𝑖 ℓ˜ 𝑖 𝑗 ∈ R 𝐵𝑟
                mi_new = np.max(np.concatenate((m_i, m_ij), axis=0))
                prev_tile_exp = np.diag(np.exp(m_i - mi_new))
                curr_tile_exp = np.diag(np.exp(m_ij - mi_new))
                li_new = np.matmul(prev_tile_exp, l_i) + \
                    np.matmul(l_ij, curr_tile_exp)
                # Write O𝑖 ← diag(ℓ new 𝑖 ) −1 (diag(ℓ𝑖)𝑒 𝑚𝑖−𝑚new 𝑖 O𝑖 + 𝑒 𝑚˜𝑖 𝑗−𝑚new 𝑖 P˜ 𝑖 𝑗V𝑗) to HBM
                li_diag = np.diag(l_i)
                li_new_diag = np.diag(li_new)
                de_normalizing_prev_computation = np.matmul(
                    li_diag, (np.dot(prev_tile_exp, O_i)))
                curr_attention = np.dot(curr_tile_exp, (np.matmul(P_ij, V_j)))
                # normalising the above two with the new li and updating the output
                self.O[i:i + self.Br] = np.divide(
                    (de_normalizing_prev_computation+curr_attention), li_new_diag)
                # updating li and mi with the new values
                self.m[i:i+self.Br] = mi_new
                self.l[i:i+self.Br] = li_new
            # end for
        # end for
        return self.O


 # let N(Sequence Length = 768)
 # let d_model(d_model = 128)
d_model = 128
N = 3072
Q = np.random.rand(N, d_model)
K = np.random.rand(N, d_model)
V = np.random.rand(N, d_model)

instance = Attention(Q, K, V)
mat1 = instance.calculate_attention()

# M = 1536
M = 1536
flashattention_instance = FlashAttention(Q, K, V, M)
mat2 = flashattention_instance.compute_flash_attention()
print(mat2)
