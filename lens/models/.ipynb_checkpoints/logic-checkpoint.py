import time

import torch

from ..utils.metrics import Metric, F1Score
from ..utils.base import NotAvailableError
from ..logic.explain import combine_local_explanations, explain_local
from ..utils.relu_nn import get_reduced_model
from .base import BaseClassifier, BaseXModel
import torch
from torch.nn.functional import one_hot
from torch import nn
import tqdm


def logit(x, tau=10):
    x = x
    return torch.log((x / (1 - x + 1e-8))+1e-8)/tau

def sigmoid(x, tau=10):
    return 1/(1+torch.exp(-tau*x))

class Phi(nn.Module):
    def __init__(self, features, calculate_entropy=False):
        super().__init__()
        self.calculate_entropy = calculate_entropy
        self.w_ = nn.Parameter(torch.Tensor(features))
        self.b = nn.Parameter(torch.Tensor(features))
        self.reset_parameters()
        self.tau = None
        self.entropy = None

    @property
    def w(self):
        return torch.exp(self.w_)
    
    @property
    def t(self):
        return -self.b/self.w

    def reset_parameters(self):
        # nn.init.constant_(self.weight, 0)
        nn.init.uniform_(self.w_, 0.1, 0.9)
        with torch.no_grad():
            self.w_.copy_(torch.log(self.w_+1e-8))
        # nn.init.uniform_(self.b, 0.1, 0.9)
        nn.init.uniform_(self.b, -0.9, -0.1)

    def forward(self, x):
        output = sigmoid(self.w*x+self.b)
        if self.tau is not None:
            output = sigmoid(self.w*x+self.b, tau=self.tau)
        if self.calculate_entropy:
            self.entropy = -(output*torch.log(output+1e-8) - (1-output)*torch.log(1-output + 1e-8)).mean()
        return output
    
class DummyPhi(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.w = torch.ones(features)
        self.b = -torch.ones(features)*0.5
        self.entropy = None
    @property
    def t(self):
        return -self.b/self.w

    def forward(self, x):
        return x
    
class LogicalLayer(nn.Module):
    def __init__(self, in_features, out_features, dummy_phi_in=False, use_weight_sigma=True, use_weight_exp=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.use_weight_sigma = use_weight_sigma
        self.use_weight_exp = use_weight_exp
        #print(dummy_phi_in)
        if dummy_phi_in:
            self.phi_in = DummyPhi(in_features)
        else:
            self.phi_in = Phi(in_features)
        #print(self.phi_in)
        if use_weight_sigma:
            self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        else: 
            self.weight_sigma = torch.ones((out_features, in_features)).float()*0.9999 # sigma(logit(0.0.9999)) = 0.9999
        
        if use_weight_exp and use_weight_sigma:
            self.weight_exp = nn.Parameter(torch.Tensor(out_features, 1))
        elif use_weight_exp and not use_weight_sigma:
            self.weight_exp = nn.Parameter(torch.Tensor(out_features, in_features))
        else: 
            self.weight_exp = torch.zeros((out_features, 1)).float() # e^0 = 1
                
        self.b = nn.Parameter(torch.Tensor(out_features))
        
        self.prune_ = nn.Parameter(torch.ones((out_features, in_features))) 
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.constant_(self.weight, 0)
        # nn.init.uniform_(self.b, 0.1, 0.9)
        nn.init.uniform_(self.b, -0.9, -0.1)
        if self.use_weight_sigma:
            nn.init.uniform_(self.weight_sigma, 0.1, 0.9)
            with torch.no_grad():
                self.weight_sigma.copy_(torch.logit(self.weight_sigma))
        if self.use_weight_exp:
            nn.init.uniform_(self.weight_exp, 0.1, 0.9)
            with torch.no_grad():
                self.weight_exp.copy_(torch.log(self.weight_exp+1e-8))

        self.set_prune(torch.ones((self.out_features, self.in_features)))

    @property
    def prune(self):
        self.prune_.requires_grad = False
        return self.prune_

    def set_prune(self, prune):
        with torch.no_grad():
            self.prune_.copy_(prune)
    
    @property
    def weight(self):
        ws = self.weight_sigma
        we = self.weight_exp
        w = sigmoid(ws)*torch.exp(we)
        w = w * self.prune
        return w

    @property
    def weight_s(self):
        ws = self.weight_sigma
        w = sigmoid(ws)
        w = w * self.prune
        return w

    @property
    def weight_e(self):
        we = self.weight_exp
        w = torch.exp(we)
        w = w * self.prune
        return w
        

    @staticmethod
    @torch.no_grad()
    def find_logic_rules(w, t_in, t_out, max_rule_len=float('inf'), max_rules=float('inf')):
        w = w.clone()
        t_in = t_in.clone()
        t_out = t_out.clone()
        t_out = t_out.item()
        ordering_scores = w
        # ordering_scores = x_train[(x_train>t_in).float()@w > t_out, :].sum(0)
        sorted_idxs = torch.argsort(ordering_scores, 0, descending=True)
        t_out -= w[t_in < 0].sum()
        mask = (t_in >= 0) & (w > 0)
        

        total_result = set()
        

        def find_logic_rules_recursive(index, current_sum):
            if len(result) > max_rules:
                return
            
            if len(current_combination) > max_rule_len:
                return
            
            if current_sum >= t_out:
                c = idxs_to_visit[current_combination].cpu().detach().tolist()
                c = tuple(sorted(c))
                result.add(c)
                # print(current_combination, 'rules', len(result))
                return


            for i in range(index, idxs_to_visit.shape[0]):
                current_combination.append(i)
                find_logic_rules_recursive(i + 1, current_sum + w[idxs_to_visit[i]])
                current_combination.pop()

        idxs_to_visit = sorted_idxs[mask[sorted_idxs]]
        current_combination = []
        result = set()
        find_logic_rules_recursive(0, 0)
        return result

    def extract_rules(self):
        ws = self.weight
        t_in = self.phi_in.t
        t_out = -self.b

        rules = []
        for i in range(self.out_features):
            w = ws[i].to('cpu')
            ti = t_in.to('cpu')
            to = t_out[i].to('cpu')
            rules.append(self.find_logic_rules(w, ti, to))

        return rules

    
    def forward(self, x):
        x = self.phi_in(torch.hstack([x, 1-x]))
        self.max_in, _ = x.max(0)
        # self.loss = -(x*torch.log(x+1e-8)).mean(-2).mean(-1)
        w = self.weight
        o = sigmoid(x @ w.t() + self.b)
        return o
    

class XLogicNN(BaseClassifier, BaseXModel):
    def __init__(self, n_classes: int, n_features: int, hidden_neurons: list, loss: torch.nn.modules.loss, dummy_phi_in=True, phi_entropy_loss=True, multi_class = False,
                 dropout_rate: 0.0 = False, l1_weight: float = 1e-4, prune_quantile=0.90, device: torch.device = torch.device('cpu'),
                 use_weight_exp = True, use_weight_sigma=True,
                 name: str = "logic.pth"):

        super().__init__(loss, name, device)
        self.activation = nn.Identity()
        self.n_classes = n_classes
        self.n_features = n_features
        self.need_pruning = True
        self.multi_class = multi_class
        self.use_weight_exp = use_weight_exp
        self.use_weight_sigma = use_weight_sigma
        self.phi_entropy_loss = phi_entropy_loss
    
        layers = []
        for i in range(len(hidden_neurons) + 1):
            input_nodes = hidden_neurons[i - 1] if i != 0 else n_features
            output_nodes = hidden_neurons[i] if i != len(hidden_neurons) else n_classes
            #print(i, (i==0 and dummy_phi_in))
            layers.append(LogicalLayer(input_nodes*2, output_nodes, dummy_phi_in = (i==0 and dummy_phi_in), use_weight_exp=use_weight_exp, use_weight_sigma=use_weight_sigma))
            layers[-1].calculate_entropy = phi_entropy_loss
           
        self.model = torch.nn.Sequential(*layers)
        self.l1_weight = l1_weight
        self.prune_quantile = prune_quantile
        self.explanations = [None for _ in range(n_classes)]

    def get_loss(self, output: torch.Tensor, target: torch.Tensor, epoch: int = None, epochs: int = None)\
            -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the is employed.
        An L1 weight regularization is also always applied

        :param epochs:
        :param epoch:
        :param output: output tensor from the forward function
        :param target: label tensor
        :return: loss tensor value
        """
        
            
        # print(output.shape)
        # print(target.shape)
        # if epoch is None or epochs is None or epoch > epochs / 2:
        l1_weight = self.l1_weight
        # else:
        #     l1_weight = self.l1_weight * 2 * epoch / epochs
        if target.shape != output.shape:
            target = one_hot(target.long(), num_classes=self.n_classes).float()
        
        output_loss = self.loss(output, target)
        reg_loss = 0
        entropy_loss = 0
        for layer in self.model:
            if self.use_weight_sigma:
                reg_loss += torch.clamp(layer.weight_s, min=1e-5).sum(-1).mean()
            else:
                reg_loss += torch.clamp(layer.weight, min=1e-5).sum(-1).mean()
            if self.phi_entropy_loss and layer.phi_in.entropy is not None:
                entropy_loss += layer.phi_in.entropy
            #reg_loss += 0.1 * (torch.clamp(layer.phi_out.b, min=1e-5)).sum(-1).mean()

        ortho_loss = 0
        if not self.multi_class:
            out_1 = output.view((*output.shape, 1))
            out_2 = output.view((*output.shape[:-1], 1, output.shape[-1]))
            ortho_loss = torch.dist(out_1*out_2, torch.eye(output.shape[-1], device=output.device).unsqueeze(0))

        
            
        
        #assert not output_loss.isnan()
        #assert not reg_loss.isnan()
        return output_loss + l1_weight * (reg_loss+ortho_loss+entropy_loss)

    def prune(self):
        """
        Prune the inputs of the model.
        """
        # self.model = prune_features(self.model, self.n_classes, self.get_device())
        for layer in self.model.children():
            if hasattr(layer, "prune"):
                layer.set_prune((layer.weight_s > torch.quantile(layer.weight_s, self.prune_quantile, dim=-1).unsqueeze(-1)).float())

    def get_global_explanation(self, x, y, target_class: int, top_k_explanations: int = None,
                               concept_names: list = None, return_time=False, simplify: bool = True,
                               metric: Metric = F1Score(), x_val=None, y_val=None, thr=0.5):
        """
        Generate a global explanation combining local explanations.

        :param y_val:
        :param x_val:
        :param metric:
        :param x: input samples
        :param y: target labels
        :param target_class: class ID
        :param top_k_explanations: number of most common local explanations to combine in a global explanation
                (it controls the complexity of the global explanation)
        :param return_time:
        :param simplify: simplify local explanation
        :param concept_names: list containing the names of the input concepts
        """
        
        def transform_to_prop_logic(symbols, rules, thresholds):
            result = []
            for rule in rules:
                elements = [f'{symbols[i]} > {thresholds[i]}' if '~' not in symbols[i] else f'{symbols[i][1:]} < {1-thresholds[i]}'  for i in rule]
                conjunction = " & ".join(elements)
                result.append('('+conjunction+')')

            prop_logic = " | ".join(result)
            return prop_logic
        
        if self.explanations[target_class] != None:
            explanation = self.explanations[target_class]
            if return_time:
                explanation, self.time
            return explanation
        
        if concept_names is None:
            concept_names = [f'x_{i}' for i in range(self.model[0].in_features//2)]
            
        start_time = time.time()
        concept_names = concept_names + [f'~{concept_name}' for concept_name in concept_names]
        #print(concept_names)
        layers_rules = [layer.extract_rules() for layer in self.model.children()]
        

        symbols = concept_names
        for layer_rules in layers_rules:
            symbols = [transform_to_prop_logic(symbols, layer_rules_i, self.model[0].phi_in.t.cpu().clone().detach().numpy().tolist()) for layer_rules_i in layer_rules]
        symbols = symbols[target_class]
        if symbols == '()':
            symbols = 'False'
        elapsed_time = time.time() - start_time
        self.explanations[target_class] = symbols
        if return_time:
            return symbols, elapsed_time
        return symbols

    
    def get_local_explanation(self, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor,
                              target_class, simplify: bool = True, concept_names: list = None, thr: float = 0.5):
        """
        Get explanation of model decision taken on the input x_sample.
,
        :param x: input samples
        :param y: target labels
        :param x_sample: input for which the explanation is required
        :param target_class: class ID
        :param simplify: simplify local explanation
        :param concept_names: list containing the names of the input concepts
        :param thr: threshold to use to select important features

        :return: Local Explanation
        """
        # return explain_local(self, x, y, x_sample, target_class, method='weights', simplify=simplify, thr=thr,
        #                      concept_names=concept_names, device=self.get_device(), num_classes=self.n_classes)
        raise NotImplementedError()


if __name__ == "__main__":
    pass
