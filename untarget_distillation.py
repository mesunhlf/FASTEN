import torch
import torch.nn as nn


def gradient_wrt_input(model, inputs, targets, criterion=nn.CrossEntropyLoss()):
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    model.zero_grad()
    loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def gradient_wrt_feature(model, source_data, target_data, layer, before_relu, criterion=nn.MSELoss()):
    source_data.requires_grad = True

    out = model.get_features(x=source_data, layer=layer, before_relu=before_relu)
    target = model.get_features(x=target_data, layer=layer, before_relu=before_relu).data.clone().detach()

    loss = criterion(out, target)
    model.zero_grad()
    loss.backward()

    data_grad = source_data.grad.data
    return data_grad.clone().detach()


def Linf_PGD(model, dat, lbl, init, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1,
             criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    # x_adv = None
    if rand_start:
        # x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        x_adv = init + dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        # x_adv = init + dat.clone().detach()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_input(model, x_adv, lbl, criterion)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad  # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad  # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    pert = x_adv - x_nat
    return x_adv.clone().detach(), pert.clone().detach()


def Linf_distillation(model, dat, init, eps, alpha, steps, layer, before_relu=True, mu=1, momentum=True,
                      rand_start=True):
    x_nat = dat.clone().detach()
    x_adv = None
    rand_start = True
    if rand_start:
        # x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        # x_adv = dat.clone().detach() + init
        x_adv = init + dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_feature(model, x_adv, dat, layer, before_relu)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            x_adv = x_adv + alpha * new_grad.sign()  # perturb the data to MAXIMIZE loss
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)

    pert = x_adv - x_nat
    return x_adv.clone().detach(), pert.clone().detach()


def gradient_wrt_feature_cosine(model, source_data, target_data, layer, before_relu, criterion=nn.CosineEmbeddingLoss()):
    source_data.requires_grad = True

    out = model.get_features(x=source_data, layer=layer, before_relu=before_relu)
    target = model.get_features(x=target_data, layer=layer, before_relu=before_relu).data.clone().detach()
    flat_out = out.view(out.shape[0], -1)
    flat_target = target.view(target.shape[0], -1)
    tar = torch.ones(len(out)).cuda()
    loss = criterion(flat_out, flat_target, target=tar)
    model.zero_grad()
    loss.backward()

    data_grad = source_data.grad.data
    return data_grad.clone().detach()

def Linf_distillation_cosine(model, dat, target, init, eps, alpha, steps, layer, before_relu=True, mu=1, momentum=True,
                      rand_start=False):
    x_nat = dat.clone().detach()
    x_adv = None
    rand_start = True
    if rand_start:
        # x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        x_adv = init + dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_feature_cosine(model, x_adv, dat, layer, before_relu)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            x_adv = x_adv + alpha * new_grad.sign()  # perturb the data to MINIMIZE loss on tgt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    pert = x_adv - x_nat
    return x_adv.clone().detach(), pert.clone().detach()

def gradient_wrt_feature_finals(model, source_data, target_data, layer, before_relu, criterion=nn.MSELoss()):
    source_data.requires_grad = True

    out = model.get_features(x=source_data, layer=layer, before_relu=before_relu)
    target, finals = model.get_features_finals(x=target_data, layer=layer, before_relu=before_relu) # .data.clone().detach()

    loss = criterion(out, target)
    model.zero_grad()
    loss.backward()

    data_grad = source_data.grad.data
    return data_grad.clone().detach(), finals.clone().detach()

#
# def gradient_wrt_feature_finals(model, source_data, target_data, layer, before_relu, criterion=nn.MSELoss()):
#     source_data.requires_grad = True
#
#     out, final = model.get_features_finals(x=source_data, layer=layer, before_relu=before_relu)
#     target = model.get_features(x=target_data, layer=layer, before_relu=before_relu).data.clone().detach()
#
#     loss = criterion(out, target)
#     model.zero_grad()
#     loss.backward()
#
#     data_grad = source_data.grad.data
#     return data_grad.clone().detach(), final.clone().detach()


def Linf_distillation_finals(model, dat, init, eps, alpha, steps, layer, before_relu=True, mu=1, momentum=True,
                      rand_start=False):
    x_nat = dat.clone().detach()
    x_adv = None
    rand_start = True
    if rand_start:
        x_adv = init + dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad, finals = gradient_wrt_feature_finals(model, x_adv, dat, layer, before_relu)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            x_adv = x_adv + alpha * new_grad.sign()  # perturb the data to MAXIMIZE loss
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)

    pert = x_adv - x_nat
    return x_adv.clone().detach(), pert.clone().detach(), finals.clone().detach()

