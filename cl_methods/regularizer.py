import torch
import torch.nn as nn
from .cl_utils import *
from copy import deepcopy


def get_regularizer(model, model_old, name, current_head, nb_class, ewc_alpha=0.9, old_state=None):
    #name = args.cl_method #cfg.TASK.CONTINUAL_METHOD
    if name == "EWC":
        fisher = old_state["fisher"] if old_state else None
        return EWC(model, model_old, current_head, nb_class, ewc_alpha, fisher)
    elif name == "SI":
        score = old_state["score"] if old_state else None
        return SI(model, model_old, current_head, nb_class, score)
    elif name == "RW":
        score = old_state["score"] if old_state else None
        fisher = old_state["fisher"] if old_state else None
        return RW(model, model_old, current_head, nb_class, ewc_alpha, score, fisher)
    else:
        #return None
        raise NotImplementedError


def _normalize_fn(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


class Regularizer:
    def update(self):
        raise NotImplementedError

    def penalty(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError


class EWC(Regularizer):
    ## From MiB github implementation
    def __init__(self, model, model_old, current_head, nb_class, ewc_alpha, fisher=None):
        self.model = model
        self.model_old = model_old
        self.alpha = ewc_alpha #cfg.TASK.EWC_ALPHA
        self.current_head = current_head #cfg.TASK.CURRENT_HEAD
        self.classes_per_task = nb_class #int(self.current_head/(self.age+1))
        self.former_head = self.current_head - self.classes_per_task

        # store old model for penalty step
        if self.model_old is not None:
            #self.model_old = model_old
            self.model_old_dict = {}
            sd = self.model_old.state_dict()
            for k, p in sd.items():
                self.model_old_dict[k] = p.cuda()
            self.penalize = True
        else:
            self.penalize = False

        # make the fisher matrix for the estimate of parameter importance
        # store the old fisher matrix (if exist) for penalize step
        if fisher is not None:
            self.fisher_old = fisher
            self.fisher = {}
            for k, p in self.fisher_old.items():
                self.fisher_old[k].require_grad = False
                self.fisher_old[k] = _normalize_fn(p)
                self.fisher_old[k] = self.fisher_old[k].cuda()
                #self.fisher[k] = torch.clone(p).cuda()
                if p.size(0) == self.former_head: # Last Linear Layer
                    if len(p.size()) > 1: # weight 
                        temp = torch.ones((self.classes_per_task,p.size(1)), requires_grad=False).cuda()
                    else: # bias
                        temp = torch.ones(self.classes_per_task, requires_grad=False).cuda()
                    self.fisher[k] = torch.cat((torch.clone(p).cuda(),temp),dim=0)
                else:
                    self.fisher[k] = torch.clone(p).cuda()
        else: # Initialization cfg.TASK == 0
            self.fisher_old = None
            self.penalize = False
            self.fisher = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad and n not in self.fisher:
                    self.fisher[n] = torch.ones_like(p, requires_grad=False).cuda()

    def update(self):
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] = (self.alpha * (p.grad ** 2)) + ((1-self.alpha) * self.fisher[n])

    def penalty(self):
        if not self.penalize:
            return torch.tensor(0.).cuda(non_blocking=True)
        else:
            loss = torch.tensor(0., requires_grad=True).cuda(non_blocking=True)
            #loss.requires_grad = True
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    #if n in self.model_old_dict and p.requires_grad:
                    if p.size(0) == self.current_head: # Last Linear Layer
                        loss += (self.fisher_old[n] * (p[:self.former_head] - self.model_old_dict[n])**2).sum()
                    else:
                        loss += (self.fisher_old[n] * (p - self.model_old_dict[n])**2).sum()
            return loss

    def get(self):
        return self.fisher

    def state_dict(self):
        state = {"name": "ewc", "fisher": self.fisher, "alpha": self.alpha,}
        return state

    def load_state_dict(self, state):
        assert (state['name'] == 'ewc'), "Error, you are trying to restore {} into ewc".format(state['name'])
        self.fisher = state["fisher"]
        for k, p in self.fisher.items():
            self.fisher[k] = p.cuda()
        self.alpha = state["alpha"]


class SI(Regularizer): # Synaptic Intelligence (Path Integral)
    ## From MiB github implementation
    def __init__(self, model, model_old, current_head, nb_class, score=None):
        self.model = model
        self.model_old = model_old
        #self.starting_new = {}
        self.current_head = current_head
        self.classes_per_task = nb_class #int(self.current_head/(cfg.TASK.AGE+1))
        self.former_head = self.current_head - self.classes_per_task

        if self.model_old is not None:
            self.penalize = True
            #self.model_old_dict = self.model_old.state_dict()
            self.model_old_dict = {}
            sd = self.model_old.state_dict()
            for k, p in sd.items():
                self.model_old_dict[k] = p.cuda()

            for k, p in self.model.named_parameters():
                #if k not in self.model_old_dict: # incremental
                if p.size(0) == self.current_head: # Last Linear Layer
                    old_p = self.model_old_dict[k].cuda()
                    new_p = p[self.former_head:].clone().detach().cuda()
                    temp_p = torch.cat((old_p,new_p),dim=0)
                    #self.starting_new[k] = p.clone().detach().cpu()
                    #new_p = torch.clone(p).detach().cuda()
                    self.model_old_dict[k] = temp_p
        else:
            self.model_old_dict = {}
            sd = deepcopy(model.state_dict())
            for k, p in sd.items():
                self.model_old_dict[k] = p.cuda()
            self.penalize = False

        if score is not None:
            self.score = score
            #self.score_actual = {}
            for n, p in score.items():
                p.requires_grad = False
                if p.size(0) == self.former_head:
                    if len(p.size()) > 1: # weight 
                        temp = torch.zeros((self.classes_per_task,p.size(1)), requires_grad=False).cuda()
                    else: # bias
                        temp = torch.zeros(self.classes_per_task, requires_grad=False).cuda()
                    temp_p = torch.cat((p.cuda(),temp),dim=0)
                    #self.score_actual[n] = _normalize_fn(temp_p.cuda())
                    self.score[n] = _normalize_fn(temp_p.cuda())
                else:
                    #self.score_actual[n] = _normalize_fn(p.cuda())
                    self.score[n] = _normalize_fn(p.cuda())

        else:
            self.score = None
            self.penalize = False

        self.delta = {n: torch.zeros_like(p, requires_grad=False).cuda()
                        for n, p in self.model.named_parameters()}
        self.model_temp = None

    def update(self):
        if self.model_temp is not None:
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    delta = p.grad.detach() * (self.model_temp[n].cuda() - p.detach())
                    self.delta[n] += delta

        self.model_temp = {k: torch.clone(p).detach().cpu()
                            for k, p in self.model.named_parameters() if p.grad is not None}

    def penalty(self):
        #print(self.penalize)
        loss = torch.tensor(0., requires_grad=True).cuda(non_blocking=True)
        if not self.penalize:
            return loss #torch.tensor(0.).cuda(non_blocking=True)
        for n, p in self.model.named_parameters():
            #loss += (self.score_actual[n] * (p - self.model_old_dict[n]).pow(2)).sum()
            if p.size(0) == self.current_head:
                loss += (self.score[n][:self.former_head] * (p[:self.former_head] - self.model_old_dict[n][:self.former_head]).pow(2)).sum()
            else:
                loss += (self.score[n] * (p - self.model_old_dict[n]).pow(2)).sum()
        return loss

    def get(self):
        score = {}
        EPS = 1e-8
        for n, p in self.model.named_parameters():
            score[n] = self.delta[n] / ((p.detach() - self.model_old_dict[n]).pow(2) + EPS)
            score[n] = torch.where(score[n] > 0, score[n], torch.tensor(0.).cuda())
            if self.score is not None and n in self.score:
                #score[n] = self.score_actual[n].cuda() + score[n]
                score[n] = self.score[n].cuda() + score[n]  # the importance is averaged
        return score  # return the score matrix

    def state_dict(self):
        state = {"name": "pi", "score": self.get(), "delta": self.delta} #,
        #        "starting_model": self.starting_new}
        return state


class RW(Regularizer):
    def __init__(self, model, model_old, cfg, score, fisher, iterations=10):
        self.model = model
        self.model_old = model_old
        self.count = 0
        self.iterations = iterations
        self.alpha = cfg.TASK.EWC_ALPHA
        self.current_head = cfg.TASK.CURRENT_HEAD
        self.classes_per_task = int(self.current_head/(cfg.TASK.AGE+1))
        self.former_head = self.current_head - self.classes_per_task

        if self.model_old is not None:
            self.model_old_dict = self.model_old.state_dict()
            self.penalize = True
            for k, p in self.model.named_parameters():
                if p.size(0) == self.current_head:
                    old_p = self.model_old_dict[k].cuda()
                    new_p = p[self.former_head:].clone().detach().cuda()
                    temp_p = torch.cat((old_p,new_p),dim=0)
                    self.model_old_dict[k] = temp_p
        else:
            self.model_old_dict = deepcopy(model.state_dict())
            self.penalize = False

        if fisher is not None and score is not None:
            self.fisher = {}
            self.score_plus_fisher = {}
            for k, p in fisher.items():
                p.requires_grad = False
                self.score_plus_fisher[k] = _normalize_fn(p.cuda())
                if p.size(0) == self.former_head:
                    if len(p.size()) > 1:
                        temp = torch.ones((self.classes_per_task,p.size(1)), requires_grad=False).cuda()
                    else:
                        temp = torch.ones(self.classes_per_task,requires_grad=False).cuda()
                    self.fisher[k] = torch.cat((torch.clone(p).cuda(),temp),dim=0)
                else:
                    self.fisher[k] = torch.clone(p).cuda()

            self.score_old = {}
            for n, p in score.items():
                p.requires_grad = False
                if p.size(0) == self.former_head:
                    if len(p.size()) > 1:
                        temp = torch.ones((self.classes_per_task,p.size(1)), requires_grad=False).cuda()
                    else:
                        temp = torch.ones(self.classes_per_task,requires_grad=False).cuda()
                    self.score_old[n] = torch.cat((torch.clone(p).cuda(),temp),dim=0)
                else:
                    self.score_old[n] = p
                self.score_plus_fisher[n] += _normalize_fn(p.cuda())
                if torch.isnan(self.score_plus_fisher[n].mean()) or torch.isinf(self.score_plus_fisher[n].mean()):
                    logger.info("Some error here")
        else:
            self.penalize = False
            self.score_old = None
            self.fisher = {}

        self.score = {n: torch.zeros_like(p, requires_grad=False).cuda()  # to compute the new score matrix
                        for n, p in self.model.named_parameters() if p.requires_grad}

        for n, p in self.model.named_parameters():  # update fisher with new keys (due to incremental classes)
            if p.requires_grad and n not in self.fisher:
                self.fisher[n] = torch.ones_like(p, requires_grad=False).cuda()

        self.model_temp = None  # to be updated at the first iteration

    def update(self):
        if self.count % self.iterations == 0:
            if self.model_temp is not None:
                # update the score
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        delta = p.grad.detach() * (self.model_temp[n].cuda() - p.detach())
                        den = 0.5 * self.fisher[n] * (p.detach() - self.model_temp[n].cuda()).pow(2) + 1e-8
                        self.score[n] += (delta / den)
            self.model_temp = {k: torch.clone(p).detach().cpu()
                                for k, p in self.model.named_parameters() if p.grad is not None}
        self.count += 1

        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] = (self.alpha * p.grad.detach().pow(2)) + ((1 - self.alpha) * self.fisher[n])

    def get_score(self):
        score = {}
        for n, p in self.score.items():
            score[n] = torch.where(p >= 0, p ,torch.tensor(0.).cuda())
            if self.score_old is not None and n in self.score_old:
                score[n] = 0.5 * (score[n] + self.score_old[n].cuda())
        return score

    def penalty(self):
        loss = torch.tensor(0., requires_grad=True).cuda(non_blocking=True)
        if not self.penalize:
            return loss # torch.tensor(0.).cuda(non_blocking=True)
        for n, p in self.model.named_parameters():
            if n in self.model_old_dict and p.requires_grad:
                if p.size(0) == self.current_head:
                    x = ((self.score_plus_fisher[n]) * (p[:self.former_head] - self.model_old_dict[n][:self.former_head]).pow(2)).sum()
                else:
                    x = ((self.score_plus_fisher[n]) * (p - self.model_old_dict[n]).pow(2)).sum()
                loss += x
        return loss

    def state_dict(self):
        state = {"name": 'rw', "score": self.get_score(), "fisher": self.fisher,
                "iteration": self.iterations, "alpha": self.alpha}
        return state


