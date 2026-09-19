"""Small independent mathematical/state fixtures for the new FD+BN integration."""
from __future__ import annotations
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch


def run_checks(legacy, exports, config):
    import torch
    import torch.nn.functional as F
    from model import DecorrelatedMLP
    from infogeo.forward_decorrelation import DenseForwardDecorrelator
    from upstream_decor_00cf470 import Decorrelator as Upstream
    from case import parameters
    from common import require
    checks = []
    gen = torch.Generator().manual_seed(811)
    x = torch.randn(12,7,generator=gen,dtype=torch.float64)+.7
    matrix = torch.eye(7,dtype=torch.float64)+.1*torch.randn(7,7,generator=gen,dtype=torch.float64)
    mean = torch.randn(7,generator=gen,dtype=torch.float64)
    upstream = Upstream(7,lr=1e-4).double()
    ours = DenseForwardDecorrelator(7,lr=1e-4,dtype=torch.float64)
    for layer in (upstream,ours):
        layer.decor_weight=matrix.clone(); layer.running_mean=mean.clone()
    a=x.clone().requires_grad_(); b=x.clone().requires_grad_()
    expected=upstream(a); actual,cache=ours(b,training=True)
    signal=torch.randn(12,7,generator=gen,dtype=torch.float64)
    gradient,=torch.autograd.grad((expected*signal).sum(),a)
    for v,w in [(actual,expected),(ours.decor_weight,upstream.decor_weight),(ours.running_mean,upstream.running_mean),
                (ours.backward(signal,cache),gradient)]:
        torch.testing.assert_close(v,w,rtol=1e-12,atol=1e-12)
    require(not torch.allclose(cache.matrix,ours.decor_weight),"fixture did not change D")
    checks.append("actual pinned upstream dense forward/state/centering derivative vs low-rank primitive; nonsymmetric old D")

    def make():
        base=legacy.ManualMLP(7,[6,4],3,seed=191,device='cpu',batchnorm=True)
        for name in exports.STATE_KEYS:
            setattr(base,name,[v.double() for v in getattr(base,name)])
        model=DecorrelatedMLP.from_base(base,1e-4)
        for layer in model.decorators:
            layer.decor_weight+=.07*torch.randn(layer.num_features,layer.num_features,generator=gen,dtype=torch.float64)
        model.training=True
        return model

    # Fully independent differentiable graph with frozen old decorator matrices.
    def graph(model,inputs):
        params=[p.detach().clone().requires_grad_() for p in parameters(model)]
        nw=len(model.weights)
        ws,bs,gs,betas=params[:nw],params[nw:2*nw],params[2*nw:2*nw+2],params[2*nw+2:]
        h=inputs
        hidden=[]; presyn=[]
        for i,(w,b,layer) in enumerate(zip(ws,bs,model.decorators)):
            u=(h-h.mean(0))@layer.decor_weight.detach().clone()
            presyn.append(u)
            z=u@w.T+b
            if i<2:
                z=(z-z.mean(0))*torch.rsqrt(z.var(0,unbiased=False)+model.bn_eps)
                h=torch.relu(gs[i]*z+betas[i]);hidden.append(h)
            else:h=z
        return h,hidden,presyn,params

    y=torch.arange(len(x))%3
    model=make(); logits,hidden,presyn,params=graph(model,x)
    expected=torch.autograd.grad(F.cross_entropy(logits,y),params)
    actual=model.bp_gradients(x,y)
    for v,w in zip([*actual.weights,*actual.biases,*actual.bn_gammas,*actual.bn_betas],expected):
        torch.testing.assert_close(v,w,rtol=2e-10,atol=2e-11)
    require(model.decorrelation_updates==[1,1,1],"BP updated decorator more than once")
    checks.append("complete BN+decorator BP chain agrees with independent autograd using old matrices")

    model=make(); reference=copy.deepcopy(model)
    logits,hidden,presyn,params=graph(reference,x)
    output_delta=(torch.softmax(logits.detach(),1)-F.one_hot(y,3))/len(x)
    feedback=[torch.randn(3,d,generator=gen,dtype=torch.float64)*.1 for d in (6,4)]
    actual=model.dfa_gradients(x,y,feedback)
    # Local surrogate for layer i: detach its incoming decorated activity;
    # inject the fixed output error at original post-ReLU hidden h_i.
    for i in range(2):
        w=params[i];b=params[3+i];g=params[6+i];beta=params[8+i]
        z=presyn[i].detach()@w.T+b
        h=torch.relu(g*(z-z.mean(0))*torch.rsqrt(z.var(0,unbiased=False)+model.bn_eps)+beta)
        gradients=torch.autograd.grad((h*(output_delta@feedback[i])).sum(),(w,b,g,beta))
        for v,wanted in zip((actual.weights[i],actual.biases[i],actual.bn_gammas[i],actual.bn_betas[i]),gradients):
            torch.testing.assert_close(v,wanted,rtol=2e-10,atol=2e-11)
    torch.testing.assert_close(actual.weights[-1],output_delta.T@presyn[-1],rtol=2e-10,atol=2e-11)
    require(model.decorrelation_updates==[1,1,1],"DFA updated decorator more than once")
    checks.append("DFA fixed pre-decorator injection matches independent layer-local autograd surrogate; classifier uses decorated inputs")

    import case as child
    small=legacy.parse_args(['--output-dir','unused','--dataset','synthetic','--batchnorm','--input-dim','8','--hidden-dims','6','4','--n-train','32','--validation-size','16','--batch-size','4','--device','cpu'])
    small_data=legacy.load_data(small)
    controlled=copy.deepcopy(config)
    controlled.update(update_budget_seconds=2.,observe_every_work_seconds=1.)
    elapsed=[0.]
    original_update=DenseForwardDecorrelator._update_state
    original_sgd=child.optimizer_step
    original_evaluate=legacy.evaluate
    def timed_decorator(*args):
        result=original_update(*args);elapsed[0]+=.25;return result
    def timed_sgd(*args):
        result=original_sgd(*args);elapsed[0]+=.25;return result
    def untimed_validation(*args):
        result=original_evaluate(*args);elapsed[0]+=100.;return result
    with patch.object(DenseForwardDecorrelator,'_update_state',timed_decorator),patch.object(child,'optimizer_step',timed_sgd),patch.object(legacy,'evaluate',untimed_validation):
        trained,_,rows,_=child.train(controlled,{'method':'fd_dfa','seed':38222,'peak_lr':.03,'decorrelation_lr':1e-5,'stage':'development'},small_data,small,legacy,exports,timer=lambda:elapsed[0])
    require([r['step'] for r in rows]==[0,1,2] and [r['training_seconds'] for r in rows]==[0.,1.,2.],"FD work was omitted or validation charged")
    require(trained.decorrelation_updates==[2,2,2] and rows[-1]['decorrelation_updates']==[2,2,2],"training/validation double-updated FD")
    checks.append("actual FD train loop charges all three decorator updates, pauses validation, and updates each decorator exactly once per step")

    # Export/restore fixture uses synthetic data, so no real dataset is touched.
    args=legacy.parse_args(['--output-dir','unused','--dataset','synthetic','--batchnorm','--input-dim','8','--hidden-dims','6','4','--n-train','32','--validation-size','16','--device','cpu'])
    data=legacy.load_data(args)
    base,fb=legacy.make_model_and_feedback(data,args,38222)
    model=DecorrelatedMLP.from_base(base,1e-5);model.training=True
    model.dfa_gradients(data.train[:8],data.labels[:8],fb)
    before=exports.model_fingerprint(model); counts=model.decorrelation_updates.copy()
    loss,accuracy=legacy.evaluate(model,data,args)
    endpoint={'method':'fd_dfa','seed':38222,'step':1,'validation_loss':loss,'validation_accuracy':accuracy}
    with tempfile.TemporaryDirectory(prefix='ndfa_fd_cpu_') as tmp:
        directory=Path(tmp)
        exports.export_final(model,data,args,endpoint,directory)
        checkpoint=torch.load(directory/'final_state.pt',weights_only=True)
        restored=exports.restore_model(checkpoint)
        saved=torch.load(directory/'final_validation.pt',weights_only=True)['logits']
        actual=exports.validation_logits(restored,data,args)
        torch.testing.assert_close(actual,saved,rtol=0,atol=0)
        require(checkpoint['schema_version']==2 and checkpoint['decorrelation_updates']==counts,"export lost decorator state")
    require(before==exports.model_fingerprint(model) and counts==model.decorrelation_updates,"validation/export mutated state")
    checks.append("export/restore bitwise validation replay with all decorator+BN states, no RNG/model mutation, no dataset access")

    base,fb=legacy.make_model_and_feedback(data,args,38223)
    base.training=True
    base.dfa_gradients(data.train[:8],data.labels[:8],fb)
    loss,accuracy=legacy.evaluate(base,data,args)
    endpoint={'method':'dfa','seed':38223,'step':1,'validation_loss':loss,'validation_accuracy':accuracy}
    with tempfile.TemporaryDirectory(prefix='ndfa_plain_cpu_') as tmp:
        directory=Path(tmp)
        exports.export_final(base,data,args,endpoint,directory)
        checkpoint=torch.load(directory/'final_state.pt',weights_only=True)
        restored=exports.restore_model(checkpoint)
        saved=torch.load(directory/'final_validation.pt',weights_only=True)['logits']
        torch.testing.assert_close(exports.validation_logits(restored,data,args),saved,rtol=0,atol=0)
        require(checkpoint['schema_version']==2 and checkpoint['model']['decorrelation'] is None and checkpoint['decorrelation_updates']==[],"plain schema2 gained FD state")
    checks.append("plain no-FD schema2 checkpoint restores and replays saved validation logits bitwise")

    # Check authoritative image inference helper against the same standardizer,
    # and show prediction independence from the chosen evaluation minibatching.
    images=torch.randint(256,(9,3,32,32),generator=gen,dtype=torch.uint8)
    base=legacy.ManualMLP(3072,[6,4],3,device='cpu',batchnorm=True)
    model=DecorrelatedMLP.from_base(base,1e-5);model.training=False
    pre={'channel_mean':torch.tensor([.4,.5,.6]),'channel_std':torch.tensor([.2,.3,.4]),'uint8_scale':255.,'validation_used':False}
    checkpoint={'preprocessing':pre}
    first=exports.inference_logits(model,images,checkpoint,batch_size=4)
    second=exports.inference_logits(model,images,checkpoint,batch_size=9)
    torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-5)
    require(model.decorrelation_updates==[0,0,0],"inference updated decorator")
    checks.append("frozen uint8 preprocessing/inference helper is batch-composition independent and leaves all state unchanged")
    return checks
