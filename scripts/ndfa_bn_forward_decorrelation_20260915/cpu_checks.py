"""Focused CPU fixtures for the new optimizer/clock/selection integration."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

from common import WorkClock, import_legacy, learning_rate, require, runtime_projection, selected_candidates
from case import make_optimizer, optimizer_step, parameters
import case as child


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    legacy,exports=import_legacy(config)
    import torch
    torch.set_num_threads(1)
    tests=[]

    clock=WorkClock(3.)
    clock.add(1.);require(not clock.done,"early budget stop")
    # Arbitrarily long validation is not an add() call and cannot consume work.
    require(clock.used==1.,"observation consumed update budget")
    clock.add(1.9);require(not clock.done,"early fractional budget stop")
    clock.add(.2);require(clock.done and abs(clock.used-3.1)<1e-12 and clock.used-3.<clock.last,"wrong overshoot")
    tests.append("budget boundary/one-update overshoot and explicit pause")

    schedule=config['schedule']
    require(abs(learning_rate(.1,0,30,schedule)-.01)<1e-12,"warmup start")
    require(abs(learning_rate(.1,1.5,30,schedule)-.1)<1e-12,"warmup boundary")
    require(abs(learning_rate(.1,30,30,schedule)-.001)<1e-12,"cosine endpoint")
    require(learning_rate(.1,31,30,schedule)==learning_rate(.1,30,30,schedule),"schedule clamp")
    vals=[learning_rate(.1,t/10,30,schedule) for t in range(15,301)]
    require(all(a>=b for a,b in zip(vals,vals[1:])),"nonmonotone cosine")
    tests.append("warmup/cosine boundaries and monotonicity")

    outcomes=[dict(c,status='complete',endpoint={'validation_loss':1.,'validation_accuracy':.5}) for c in config['development']]
    chosen,_=selected_candidates(config,outcomes)
    require(all(r['peak_lr']==min(c['peak_lr'] for c in config['candidates'][m]) and r['boundary_selected'] for m,r in chosen.items()),"selection LR tie break")
    for row in outcomes:
        if row['candidate_id']=='c04':row['endpoint']['validation_accuracy']=.6
    chosen,_=selected_candidates(config,outcomes)
    require(all(r['peak_lr']==.1 and not r['boundary_selected'] for r in chosen.values()),"selection accuracy tie break")
    outcomes[0]['status']='numerical_failed'
    chosen,scores=selected_candidates(config,outcomes)
    require(not next(r for r in scores if r['method']==outcomes[0]['method'] and r['candidate_id']==outcomes[0]['candidate_id'])['valid'],"failed candidate stayed eligible")
    try:selected_candidates(config,outcomes[:-1])
    except ValueError:pass
    else:raise AssertionError("selection accepted missing development case")
    tests.append("selection complete inventory, two-seed failures, ties and boundary flag")

    benchmark=[{'status':'complete','invocation_wall_seconds':15.,'endpoint':{'training_seconds':2.,'step':500},
                'validation_count':2,'evaluation_seconds':.1} for _ in range(4)]
    gate=runtime_projection(config,benchmark,45.)
    require(gate['accepted'] and not runtime_projection(config,benchmark,45.,1000.)['accepted'],"resource gate ignores allocation end")
    tests.append("full104-case overhead/evaluation projection and Slurm remaining-time limit")

    small=legacy.parse_args(['--output-dir','unused','--dataset','synthetic','--batchnorm',
                            '--hidden-dims','8','4','--input-dim','8','--n-train','32',
                            '--validation-size','16','--batch-size','4','--steps','2',
                            '--device','cpu','--feedback-scale','.1','--relative-damping','30'])
    data=legacy.load_data(small)
    controlled=copy.deepcopy(config)
    controlled.update(update_budget_seconds=2.,observe_every_work_seconds=1.)
    fixture_case={'method':'dfa','seed':5,'peak_lr':.03,'stage':'development'}
    elapsed=[0.]
    original_evaluate=legacy.evaluate
    original_step=child.optimizer_step
    def delayed_evaluate(*args):
        result=original_evaluate(*args)
        elapsed[0]+=100. # Simulated diagnostic/validation delay, excluded from work.
        return result
    def timed_optimizer(*args):
        result=original_step(*args)
        elapsed[0]+=1.
        return result
    with patch.object(legacy,'evaluate',side_effect=delayed_evaluate),patch.object(child,'optimizer_step',side_effect=timed_optimizer):
        _,_,rows,_=child.train(controlled,fixture_case,data,small,legacy,exports,timer=lambda:elapsed[0])
    require([r['step'] for r in rows]==[0,1,2] and [r['training_seconds'] for r in rows]==[0.,1.,2.],"train clock counted validation or stopped at wrong boundary")
    require(elapsed[0]==302. and rows[-1]['observation_reason']=='work_checkpoint+final',"train fixture did not exercise pauses/endpoints")
    tests.append("actual train loop excludes300 simulated validation seconds and stops at2 update seconds")
    progress={};count=[0];elapsed[0]=0.
    def failing_optimizer(*args):
        count[0]+=1
        if count[0]==2:raise FloatingPointError('fixture numerical failure')
        return timed_optimizer(*args)
    with patch.object(legacy,'evaluate',side_effect=delayed_evaluate),patch.object(child,'optimizer_step',side_effect=failing_optimizer):
        try:child.train(controlled,fixture_case,data,small,legacy,exports,timer=lambda:elapsed[0],progress=progress)
        except FloatingPointError:pass
        else:raise AssertionError('injected numerical failure not propagated')
    require([r['step'] for r in progress['rows']]==[0,1] and progress['clock'].used==1. and 'model' in progress and 'optimizer' in progress,"finite failure prefix/context lost")
    tests.append("actual train loop retains finite observation prefix and diagnostic objects on failure")
    for method in ('bp','dfa','ndfa'):
        model,feedback=legacy.make_model_and_feedback(data,small,1)
        reference,reference_feedback=legacy.make_model_and_feedback(data,small,1)
        plain=copy.deepcopy(config);plain['optimizer'].update(momentum=0.,weight_decay=0.)
        optimizer=make_optimizer(model,plain,.03)
        generator=torch.Generator().manual_seed(40001)
        sampler=torch.Generator().manual_seed(30001)
        for _ in range(2):
            idx=torch.randint(len(data.train),(small.batch_size,),generator=sampler)
            x,y=legacy.paired_batch(data,idx,small,generator)
            model.training=reference.training=True
            raw=model.bp_gradients(x,y) if method=='bp' else model.dfa_gradients(x,y,feedback)
            expected_raw=reference.bp_gradients(x,y) if method=='bp' else reference.dfa_gradients(x,y,reference_feedback)
            update=legacy.condition(model,raw,x,method,small)
            expected=legacy.condition(reference,expected_raw,x,method,small)
            if method=='ndfa':
                for before,after in zip(raw.weights[:-1],update.weights[:-1]):
                    torch.testing.assert_close(before.norm(),after.norm(),rtol=1e-5,atol=1e-7)
            optimizer_step(model,update,optimizer,.03)
            reference.apply_gradients(expected,lr=.03)
            for actual,wanted in zip(parameters(model),parameters(reference)):
                torch.testing.assert_close(actual,wanted,rtol=2e-5,atol=2e-7)
            for a,b in zip(model.bn_running_mean,reference.bn_running_mean):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)
        tests.append(method+" zero-momentum/no-L2 matches archived two-step update and BN semantics")

    # Independent hand recurrence includes evolving weight decay, BN/bias exclusions,
    # and first-buffer behavior. Gradient entries differ to catch mapping swaps.
    model,_=legacy.make_model_and_feedback(data,small,2)
    optimizer=make_optimizer(model,config,.02)
    expected=[p.clone() for p in parameters(model)]
    buffers=[None]*len(expected)
    for iteration in range(2):
        gradients=[torch.full_like(p,(i+1)*.01+iteration*.03) for i,p in enumerate(parameters(model))]
        nw,nb,ng=len(model.weights),len(model.biases),len(model.bn_gamma)
        update=legacy.Gradients(gradients[:nw],gradients[nw:nw+nb],[],0.,gradients[nw+nb:nw+nb+ng],gradients[nw+nb+ng:])
        for i,(p,g) in enumerate(zip(expected,gradients)):
            direction=g+(config['optimizer']['weight_decay']*p if i<nw else 0.)
            buffers[i]=direction.clone() if buffers[i] is None else .9*buffers[i]+direction
            expected[i]=p-.02*buffers[i]
        optimizer_step(model,update,optimizer,.02)
        for actual,wanted in zip(parameters(model),expected):
            torch.testing.assert_close(actual,wanted,rtol=1e-6,atol=1e-7)
    tests.append("manual two-step momentum/L2 recurrence, first buffer, and all parameter groups")

    states=[];prefixes=[]
    from experiments.run_ndfa_bn_confirmation_case import model_fingerprint as supervised_fingerprint
    for method in config["methods"]:
        model,_=legacy.make_model_and_feedback(data,small,36000)
        if method == "fd_dfa":
            from model import DecorrelatedMLP
            model=DecorrelatedMLP.from_base(model,1e-5)
        states.append(supervised_fingerprint(model))
        generator=torch.Generator().manual_seed(40000+36000)
        sampler=torch.Generator().manual_seed(30000+36000)
        records=[]
        for _ in range(16):
            idx=torch.randint(len(data.train),(small.batch_size,),generator=sampler)
            x,y=legacy.paired_batch(data,idx,small,generator)
            records.append((exports.tensor_digest(idx),exports.tensor_digest(x),exports.tensor_digest(y)))
        prefixes.append(records)
    require(all(s==states[0] for s in states) and all(p==prefixes[0] for p in prefixes),"paired initialization/view prefix differs")
    images=torch.randint(256,(16,3,32,32),generator=torch.Generator().manual_seed(9182),dtype=torch.uint8)
    image_data=legacy.Data(images,torch.arange(16)%10,images[:4],torch.arange(4),3072,10,
                          channel_mean=torch.tensor([.4,.5,.6]),channel_std=torch.tensor([.2,.2,.2]))
    image_args=copy.copy(small);image_args.dataset='cifar10'
    image_prefixes=[]
    for _ in range(3):
        sampler=torch.Generator().manual_seed(66000);views=torch.Generator().manual_seed(76000)
        digest=[]
        for _ in range(16):
            idx=torch.randint(16,(4,),generator=sampler);x,y=legacy.paired_batch(image_data,idx,image_args,views)
            digest.append((exports.tensor_digest(idx),exports.tensor_digest(x),exports.tensor_digest(y)))
        image_prefixes.append(digest)
    require(image_prefixes[0]==image_prefixes[1]==image_prefixes[2],"CIFAR crop/flip paired prefixes differ")
    from fd_checks import run_checks
    tests.extend(run_checks(legacy,exports,config))
    require(not torch.cuda.is_initialized(),"CPU fixtures initialized CUDA")
    tests.append("paired initialization and16 augmented minibatch prefixes; no CUDA")
    print(json.dumps({'accepted':True,'tests':tests,'count':len(tests),'torch':str(torch.__version__),'cuda_initialized':False},indent=2))


if __name__=='__main__':
    main()
