from argparse import ArgumentParser

import torch
import sklearn
import torch.nn as nn

from utils import init_featurizer, mkdir_p, get_configure, load_model, load_dataloader, predict, get_reactive_template_labels

def mask_loss(loss_criterion1, loss_criterion2, pred_v, pred_r, true_v, true_r, vmask, rmask):
    vmask, rmask = vmask.double, rmask.double
    vloss = (loss_criterion1(pred_v, true_v) * (vmask != 0)).float().mean()
    rloss = (loss_criterion2(pred_r, true_r) * (rmask != 0)).float().mean()
    return vloss + rloss

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterions, optimizer):
    model.train()
    train_R_loss = 0
    train_T_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, adm_lists, bonds_dicts, true_VT, true_RT, masks = batch_data
        if len(smiles) == 1:
            print ('Skip problematic graph')
            continue
        pred_VT, pred_RT, pred_VR, pred_RR, pred_VI, pred_RI, _ = predict(args, model, bg, adm_lists, bonds_dicts)
        true_VT, true_VR, mask_V = get_reactive_template_labels(true_VT, masks, pred_VI)
        true_RT, true_RR, mask_R = get_reactive_template_labels(true_RT, masks, pred_RI)
        true_VT, true_RT, true_VR, true_RR, mask_V, mask_R = true_VT.to(args['device']), true_RT.to(args['device']), true_VR.to(args['device']), true_RR.to(args['device']), mask_V.to(args['device']), mask_R.to(args['device'])
        
        R_loss = mask_loss(loss_criterions[0], loss_criterions[0], pred_VR, pred_RR, true_VR, true_RR, mask_V, mask_R)
        T_loss = mask_loss(loss_criterions[1], loss_criterions[2], pred_VT, pred_RT, true_VT, true_RT, mask_V, mask_R)
        loss = R_loss + T_loss
        train_R_loss += R_loss.item()
        train_T_loss += T_loss.item()
        optimizer.zero_grad()      
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()
        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, reactive loss %.4f, template loss %.4f' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), R_loss.item(), T_loss.item()), end='', flush=True)

    print('\nepoch %d/%d, train reactive loss %.4f, template loss %.4f' % (epoch + 1, args['num_epochs'], train_R_loss/batch_id, train_T_loss/batch_id))

def run_an_eval_epoch(args, model, data_loader, loss_criterions):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, adm_lists, bonds_dicts, true_VT, true_RT, masks = batch_data
            if len(smiles) == 1:
                print ('Skip problematic graph')
                continue
            pred_VT, pred_RT, pred_VR, pred_RR, pred_VI, pred_RI, _ = predict(args, model, bg, adm_lists, bonds_dicts)
            true_VT, true_VR, mask_V = get_reactive_template_labels(true_VT, masks, pred_VI)
            true_RT, true_RR, mask_R = get_reactive_template_labels(true_RT, masks, pred_RI)
            true_VT, true_RT, true_VR, true_RR, mask_V, mask_R = true_VT.to(args['device']), true_RT.to(args['device']), true_VR.to(args['device']), true_RR.to(args['device']), mask_V.to(args['device']), mask_R.to(args['device'])
            loss = mask_loss(loss_criterions[1], loss_criterions[2], pred_VT, pred_RT, true_VT, true_RT, mask_V, mask_R)
            val_loss += loss.item()
    return val_loss/batch_id


def main(args):
    if args['model_name'] == 'default':
        if args['sep']:
            args['model_name'] = 'LocalTransform_sep.pth'
        else:
            args['model_name'] = 'LocalTransform_mix.pth'
    else:
        args['model_name'] = '%s.pth' % args['model_name']
        
    args['model_path'] = '../models/' + args['model_name']
    args['config_path'] = '../data/configs/%s' % args['config']
    args['data_dir'] = '../data/%s' % args['dataset']
    mkdir_p('../models')                          
    args = init_featurizer(args)
    model, loss_criterions, optimizer, scheduler, stopper = load_model(args)   
    train_loader, val_loader, test_loader = load_dataloader(args)
    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterions, optimizer)
        val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterions)
        early_stop = stopper.step(val_loss, model) 
        scheduler.step()
        print('epoch %d/%d, validation loss: %.4f' %  (epoch + 1, args['num_epochs'], val_loss))
        print('epoch %d/%d, Best loss: %.4f' % (epoch + 1, args['num_epochs'], stopper.best_score))
        if early_stop:
            print ('Early stopped!!')
            break

    stopper.load_checkpoint(model)
    test_loss = run_an_eval_epoch(args, model, test_loader, loss_criterions)
    print('test loss: %.4f' % test_loss)
    
if __name__ == '__main__':
    parser = ArgumentParser('Training arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_480k', help='Dataset to use')
    parser.add_argument('-c', '--config', default='default_config', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', default=16, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=20, help='Maximum number of epochs for training')
    parser.add_argument('-m', '--model-name', type=str, default='default', help='Model name')
    parser.add_argument('-p', '--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('-w', '--negative-weight', type=float, default=0.5, help='Loss weight for negative labels')
    parser.add_argument('-s', '--sep', default=False, help='Train the model with reagent seperated or not')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule-step', type=float, default=6, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=20, help='Print the training progress every X mini-batches')
    args = parser.parse_args().__dict__
    args['mode'] = 'train'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'], 'seperate reagent: %s' % args['sep'])
    main(args)