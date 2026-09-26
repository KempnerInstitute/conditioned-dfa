"""Manual tanh/sigmoid classifier for stronger DFA-Stall baseline controls.

Architecture, Xavier initialization distribution and summed one-vs-rest binary
loss match the DFA-Stall setting. BCE is evaluated in logit space for numerical
stability. New paired seeds are not bitwise reruns of historical cohorts.
"""
import torch
import torch.nn.functional as F
from .dfa import ManualMLP,Gradients


class TanhSigmoidMLP(ManualMLP):
    def __init__(self,input_dim,hidden_dims,output_dim,*,seed=0,device="cpu"):
        super().__init__(input_dim,hidden_dims,output_dim,seed=seed,device=device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            dims=[input_dim,*hidden_dims,output_dim]
            layers=[torch.nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:])]
            for layer in layers:
                torch.nn.init.xavier_uniform_(layer.weight);torch.nn.init.zeros_(layer.bias)
            self.weights=[layer.weight.detach().to(device).clone() for layer in layers]
            self.biases=[layer.bias.detach().to(device).clone() for layer in layers]

    def forward(self,x):
        h=x.to(self.device);activities=[h];pre=[]
        for index,(w,b) in enumerate(zip(self.weights,self.biases)):
            a=h@w.T+b;pre.append(a)
            h=a.tanh() if index<self.n_hidden_layers else a
            activities.append(h)
        self.last_activities=activities
        return h,activities,pre

    def _gradients(self,x,y,feedback):
        logits,a,_=self.forward(x)
        target=F.one_hot(y.to(self.device),self.output_dim).to(logits.dtype)
        loss=F.binary_cross_entropy_with_logits(logits,target,reduction="sum")/len(x)
        output=(logits.sigmoid()-target)/len(x)
        deltas=[None]*len(self.weights);deltas[-1]=output
        for i in reversed(range(self.n_hidden_layers)):
            signal=output@feedback[i] if feedback is not None else deltas[i+1]@self.weights[i+1]
            deltas[i]=signal*(1-a[i+1].square())
        return Gradients([d.T@h for d,h in zip(deltas,a)], [d.sum(0) for d in deltas],deltas,float(loss))

    def bp_gradients(self,x,y):return self._gradients(x,y,None)
    def dfa_gradients(self,x,y,feedback):return self._gradients(x,y,feedback)
