import torch
from torch.autograd import Variable
import math

class NCELoss:
    
    def __init__(self,noise,noise_ratio = 25):
        self.noise = noise        
        self.noise_ratio = noise_ratio
        
    
    def forward(self,input,target,noise_samples = None):
        '''
        input = embedding/output of the decoder rnn 
        target = index of the target word in the vocabulary
        '''
        
        if noise_samples:
            self.noise_samples = noise_samples
        else:
            noise_samples = torch.multinomial(
                    self.noise,
                    self.noise_ratio,
                    replacement = True
                    ).view(1,-1)
        
        e = 2.71
        input = input.view(-1,1)
        p_target = e**(input[target.data[0]])
        #print('p_target', p_target)
        p_noise = [e**(input[noise_idx]) for noise_idx in noise_samples]
        
        data_prob = torch.log(p_target/(p_target+ self.noise_ratio*self.noise[target.data[0]]))
        
        #noise_prob = sum([torch.log(1-(p_noise[0][p]/(p_noise[0][p]+self.noise_ratio*self.noise[noise_samples[0][p]]))) for p in range(self.noise_ratio)])
        noise_prob = 0
        for p in range(self.noise_ratio):
            noise_prob+= torch.log(1-(p_noise[0][p]/(p_noise[0][p]+self.noise_ratio*self.noise[noise_samples[0][p]])))
        #print(data_prob)
        #print(noise_prob)
        objective_function = -(data_prob + noise_prob)
        '''
        data_prob = torch.log((1/(1+self.noise_ratio))*p_target)
        j = self.noise_ratio/(1+self.noise_ratio)
        noise_prob = sum([torch.log(j*Variable(torch.FloatTensor([self.noise[noise_samples[0][p]]]))) for p in range(self.noise_ratio)])
        objective_function = -(data_prob+noise_prob)
        '''
        return objective_function
        