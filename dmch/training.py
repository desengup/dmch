import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from . import from_inputs, to_inputs

def _calc_utility(inputs, mechanism, values, output_components=False):
    allocation, payment = mechanism(inputs)
    if output_components:
        return allocation*values-payment, allocation, payment
    else:
        return allocation*values-payment
    
def _calc_regret(inputs, mechanism, misreport, bidders, leaky=False):
    values, context = from_inputs(inputs, bidders)
    u_true, a_true, p_true = _calc_utility(inputs, mechanism, values, output_components=True)
    regret = torch.zeros(values.shape).to(values.get_device())
    for bidder in range(bidders):
        bidder_mask = torch.zeros(values.shape).to(values.get_device())
        bidder_mask[:,bidder] = 1.0
        response = misreport * bidder_mask + values * (1-bidder_mask)
        u_response = _calc_utility(to_inputs(response,context), mechanism, values)
        if leaky:
            regret = regret + F.leaky_relu((u_response - u_true) * bidder_mask)
        else:
            regret = regret + F.relu((u_response - u_true) * bidder_mask)
    return regret.mean(dim=0)

def _best_misreport(values, inputs, mechanism, bidders, device, misreport_lr, misreport_epochs):
    misreport = (torch.FloatTensor(values.shape).uniform_(0, 1).to(device) * values).detach().requires_grad_(True)
    misreport_optimizer = optim.Adam([misreport], lr=misreport_lr)
            
    mechanism.eval()
    for _ in range(misreport_epochs):
        misreport_optimizer.zero_grad()
        regret = _calc_regret(
            inputs, 
            mechanism,
            misreport,
            bidders,
            leaky=True)
        (-regret.sum()).backward()
        nn.utils.clip_grad_norm_([misreport], 1.0)
        misreport_optimizer.step()
    mechanism.train()
    return misreport.detach().clone().requires_grad_(False)

def train(mechanism, values_loader, bidders, **kwargs):
    # load parameters
    device = kwargs['device'] if 'device' in kwargs else 'cpu'
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 1
    rho = kwargs['rho'] if 'rho' in kwargs else 100
    mechanism_lr = kwargs['mechanism_lr'] if 'mechanism_lr' in kwargs else 1e-3
    misreport_lr = kwargs['misreport_lr'] if 'misreport_lr' in kwargs else 1e-3
    misreport_epochs = kwargs['misreport_epochs'] if 'misreport_epochs' in kwargs else 10
    consider_dsic = kwargs['consider_dsic'] if 'consider_dsic' in kwargs else True
    consider_ir = kwargs['consider_ir'] if 'consider_ir' in kwargs else True
    
    # Initialize augmented lagrangian parameters
    if consider_dsic:
        lambda_dsic = torch.zeros(bidders).to(device)
    if consider_ir:
        lambda_ir   = torch.zeros(bidders).to(device)

    # Initalize the optimizer
    mechanism_optimizer = optim.Adam(mechanism.parameters(), lr=mechanism_lr)
    
    report_data = []

    for epoch in tqdm(range(epochs)):
        for batch_idx,(values_list) in enumerate(values_loader):
            inputs = values_list[0].to(device)
            values, context = from_inputs(inputs, bidders)
            
            if consider_dsic:
                misreport = _best_misreport(
                    values, 
                    inputs, 
                    mechanism, 
                    bidders, 
                    device, 
                    misreport_lr, 
                    misreport_epochs)

            # Start the gradient computation
            mechanism.zero_grad()
            
            # calculate the utilities and prices
            utility, allocation, payment = _calc_utility(
                inputs, mechanism, values, output_components=True)
    
            if consider_dsic:
                # compute expected regret 
                dsic_violation = _calc_regret(
                    inputs,
                    mechanism,
                    misreport,
                    bidders)
            
            if consider_ir:
                # compute individual rationality violation
                ir_violation = F.relu(-utility).mean(dim=0)
    
            # compute components of the loss function
            revenue = payment.sum(dim=-1).mean()
            
            if consider_dsic:
                total_dsic_violation = dsic_violation.sum()
            
            if consider_ir:
                total_ir_violation = ir_violation.sum()
            
            total_violation = 0
            if consider_dsic:
                total_violation += total_dsic_violation.pow(2)
                
            if consider_ir:
                total_violation += total_ir_violation.pow(2)
                
            # define the loss 
            loss = -revenue+0.5*rho*(total_violation)
            
            if consider_dsic:
                loss += (lambda_dsic*dsic_violation).sum()
                
            if consider_ir:
                loss += (lambda_ir*ir_violation).sum()
                
            
            # Trigger the autogradient calculation
            loss.backward()
        
            # Clip the norm to prevent exploding gradients
            nn.utils.clip_grad_norm_(mechanism.parameters(), 1.0)
            
            # Take a step towards the gradient
            mechanism_optimizer.step()
            
            mechanism.eval()
            
            if consider_dsic:
                misreport = _best_misreport(
                    values, 
                    inputs, 
                    mechanism, 
                    bidders, 
                    device, 
                    misreport_lr, 
                    misreport_epochs)
            
                # Update the augmented lagrangian parameters
                dsic_violation_next = _calc_regret(
                    inputs,
                    mechanism,
                    misreport,
                    bidders)
                lambda_dsic = (lambda_dsic + rho * dsic_violation_next).detach()
            
            if consider_ir:
                u_next = _calc_utility(inputs, mechanism, values)
                ir_violation_next = F.relu(-u_next).mean(dim=0)
            
                lambda_ir = (lambda_ir + rho * ir_violation_next).detach()
            
            mechanism.train()
            
            report_item = {
                'epoch': epoch,
                'batch': batch_idx,
                'revenue':revenue.item(),
                'loss':loss.item()}
            
            if consider_dsic:
                report_item['total_dsic_violation']=total_dsic_violation.item()
            
            if consider_ir:
                report_item['total_ir_violation']=total_ir_violation.item()
                
            report_data.append(report_item)
            
    return report_data

def evaluate(mechanism, inputs_loader, bidders, **kwargs):
    # load parameters
    device = kwargs['device'] if 'device' in kwargs else 'cpu'
    misreport_lr = kwargs['misreport_lr'] if 'misreport_lr' in kwargs else 1e-3
    misreport_epochs = kwargs['misreport_epochs'] if 'misreport_epochs' in kwargs else 10
    consider_dsic = kwargs['consider_dsic'] if 'consider_dsic' in kwargs else True
    consider_ir = kwargs['consider_ir'] if 'consider_ir' in kwargs else True
    
    report_data = []

    mechanism.eval()
    for batch_idx,(input_list) in enumerate(inputs_loader):
        inputs = input_list[0].to(device)
        values, context = from_inputs(inputs, bidders)
        
        if consider_dsic:
            misreport = _best_misreport(
                values, 
                inputs, 
                mechanism, 
                bidders, 
                device, 
                misreport_lr, 
                misreport_epochs)
            
        # calculate the utilities and prices
        utility, allocation, payment = _calc_utility(
            inputs, mechanism, values, output_components=True)
    
        if consider_dsic:
            # compute expected regret 
            dsic_violation = _calc_regret(
                inputs,
                mechanism,
                misreport,
                bidders)
    
        if consider_ir:
            # compute individual rationality violation
            ir_violation = F.relu(-utility).mean(dim=0)
    
        # compute components of the loss function
        revenue = payment.sum(dim=-1).mean()
        
        if consider_dsic:
            total_dsic_violation = dsic_violation.sum()
        
        if consider_ir:
            total_ir_violation = ir_violation.sum()
        
        report_item = {
            'batch': batch_idx,
            'revenue':revenue.item(),
        }
        
        if consider_dsic:
            report_item['total_dsic_violation']=total_dsic_violation.item()
            
        if consider_ir:
            report_item['total_ir_violation']=total_ir_violation.item()
            
        report_data.append(report_item)
        
    mechanism.train()        
    return report_data


