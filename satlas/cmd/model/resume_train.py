import argparse
import json
import numpy as np
import os
import sys
import time
import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
import satlas.model.dataset
import satlas.model.evaluate
import satlas.model.models
import satlas.model.util
import satlas.transforms
import pandas as pd

def make_warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor, warmup_delay=0):
    def f(x):
        if x < warmup_delay:
            return 1
        if x >= warmup_delay + warmup_iters:
            return 1
        alpha = float(x - warmup_delay) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def save_atomic(state_dict, dir_name, fname):
    tmp_fname = fname + '.tmp'
    torch.save(state_dict, os.path.join(dir_name, tmp_fname))
    os.rename(os.path.join(dir_name, tmp_fname), os.path.join(dir_name, fname))

def load_checkpoint(model, optimizer, scaler, filepath):
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in state_dict:
        scaler.load_state_dict(state_dict['scaler_state_dict'])
    model_state_dict = state_dict['model_state_dict']
    epoch = state_dict['epoch']
    summary_epoch = state_dict['summary_epoch']
    train_loss = state_dict['train_loss']
    train_task_losses = state_dict['train_task_losses']
    val_loss = state_dict['val_loss']
    val_task_losses = state_dict['val_task_losses']
    val_score = state_dict['val_score']
    best_score = state_dict['best_score']
    val_scores = state_dict['val_scores']
    
    return model, optimizer, model_state_dict, scaler, epoch, summary_epoch, train_loss, train_task_losses, val_loss, val_task_losses, val_score, val_scores, best_score

def main(args, config):
    rank = args.local_rank
    primary = rank is None or rank == 0
    is_distributed = rank is not None

    channels = config.get('Channels', ['tci', 'fake', 'fake'])
    batch_size = config['BatchSize'] // args.world_size
    val_batch_size = config.get('ValBatchSize', config['BatchSize'])

    # Set Task info if needed.
    for spec in config['Tasks']:
        if 'Task' not in spec:
            spec['Task'] = satlas.model.dataset.tasks[spec['Name']]
    if 'ChipSize' in config:
        satlas.model.dataset.chip_size = config['ChipSize']

    train_transforms = satlas.transforms.get_transform(config, config.get('TrainTransforms', [{
        'Name': 'CropFlip',
        'HorizontalFlip': True,
        'VerticalFlip': True,
        'Crop': 256,
    }]))
    batch_transform = satlas.transforms.get_batch_transform(config, config.get('TrainBatchTransforms', []))

    def get_task_transforms(k):
        task_transforms = {}
        for spec in config['Tasks']:
            if k not in spec:
                continue
            task_transforms[spec['Name']] = satlas.transforms.get_transform(config, spec[k])
        if len(task_transforms) == 0:
            return None
        return task_transforms

    # Load train and validation data.
    train_data = satlas.model.dataset.Dataset(
        task_specs=config['Tasks'],
        transforms=train_transforms,
        channels=channels,
        max_tiles=config.get('TrainMaxTiles', None),
        num_images=config.get('NumImages', 1),
        task_transforms=get_task_transforms('TrainTransforms'),
        phase="Train",
        custom_images=config.get('CustomImages', False),
    )

    val_data = satlas.model.dataset.Dataset(
        task_specs=config['Tasks'],
        transforms=satlas.transforms.get_transform(config, config.get('ValTransforms', [])),
        channels=channels,
        max_tiles=config.get('ValMaxTiles', None),
        num_images=config.get('NumImages', 1),
        task_transforms=get_task_transforms('ValTransforms'),
        phase="Val",
        custom_images=config.get('CustomImages', False),
    )

    print('loaded {} train, {} valid'.format(len(train_data), len(val_data)))

    train_sampler_cfg = config.get('TrainSampler', {'Name': 'random'})
    if train_sampler_cfg['Name'] == 'random':
        train_sampler = torch.utils.data.RandomSampler(train_data)
    elif train_sampler_cfg['Name'] == 'tile_weight':
        with open(train_sampler_cfg['Weights'], 'r') as f:
            tile_weights = json.load(f)
        train_sampler = train_data.get_tile_weight_sampler(tile_weights=tile_weights)
    else:
        raise Exception('unknown train sampler {}'.format(train_sampler_cfg['Name']))

    val_sampler = torch.utils.data.SequentialSampler(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config.get('NumLoaderWorkers', 4),
        collate_fn=satlas.model.util.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=val_batch_size,
        sampler=val_sampler,
        num_workers=config.get('NumLoaderWorkers', 4),
        collate_fn=satlas.model.util.collate_fn,
    )

    # Initialize torch distributed.
    if is_distributed:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Initialize model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config['Model']
    model = satlas.model.models.get_model({
        'config': model_config,
        'channels': channels,
        'tasks': config['Tasks'],
    })

    # Construct optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_config = config['Optimizer']
    if optimizer_config['Name'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=optimizer_config['InitialLR'])
    elif optimizer_config['Name'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=optimizer_config['InitialLR'])

    # Half-precision stuff.
    half_enabled = config.get('Half', False)
    scaler = torch.cuda.amp.GradScaler(enabled=half_enabled)
    print(f"Scaler state: {scaler.get_scale()}")

    # Load model if requested.
    resume_path = config.get('ResumePath', None)
    if resume_path:
        if primary: 
            print('resuming from', resume_path)

        model, optimizer, model_state_dict, scaler, list_epoch, list_summary_epoch, list_train_loss, list_train_task_losses, list_val_loss, list_val_task_losses, list_val_score, list_val_scores, list_best_score = load_checkpoint(model, optimizer, scaler, resume_path)
        best_score = max([x for x in list_best_score if x is not None])
        summary_epoch = list_summary_epoch[-1]
        start_epoch = list_epoch[-1] + 1
        if primary:
            print('Resuming training from epoch', start_epoch)
            
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        if primary and (missing_keys or unexpected_keys):
            print('missing={}; unexpected={}'.format(missing_keys, unexpected_keys))

    # Move model to the correct device.
    model = model.to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    parameter_list = []
    layer_summaries = {
        "Backbone": 0,
        "Intermediates": 0,
        "Heads": 0
    }
    # # Duyệt qua các tham số
    # for name, param in model.named_parameters():
    #     parameter_list.append({
    #         "Layer": name,
    #         "Shape": list(param.shape),
    #         "Number of Parameters": param.numel(),
    #         "Device": param.device
    #     })
    
    # Chuyển đổi sang bảng pandas để hiển thị gọn hơn
    # Phân loại tham số dựa trên tên
    for name, param in model.named_parameters():
        if "backbone" in name:
            layer_summaries["Backbone"] += param.numel()
        elif "intermediate" in name:
            layer_summaries["Intermediates"] += param.numel()
        elif "head" in name:
            layer_summaries["Heads"] += param.numel()
    
    # Chuyển kết quả sang dataframe để in ra
    df = pd.DataFrame([
        {"Layer Type": key, "Total Parameters": value}
        for key, value in layer_summaries.items()
    ])
    print(df)

    # Prepare save directory.
    save_path = config['SavePath']
    save_path = save_path.replace('LABEL', os.path.basename(args.config_path).split('.')[0])
    if primary:
        os.makedirs(save_path, exist_ok=True)
        print('saving to', save_path)

    # Freeze parameters if desired.
    unfreeze_iters = None
    if 'Freeze' in config:
        freeze_prefixes = config['Freeze']
        freeze_params = []
        for name, param in model.named_parameters():
            should_freeze = False
            for prefix in freeze_prefixes:
                if name.startswith(prefix):
                    should_freeze = True
                    break
            if should_freeze:
                # if primary: print('freeze', name)
                param.requires_grad = False
                freeze_params.append((name, param))
        if 'Unfreeze' in config:
            unfreeze_iters = config['Unfreeze'] // batch_size // args.world_size
            def unfreeze_hook():
                for name, param in freeze_params:
                    # if primary: print('unfreeze', name)
                    param.requires_grad = True

    # Configure learning rate schedulers.
    if 'WarmupExamples' in config:
        warmup_iters = config['WarmupExamples'] // batch_size // args.world_size
        warmup_delay_iters = config.get('WarmupDelay', 0) // batch_size // args.world_size
        warmup_lr_scheduler = make_warmup_lr_scheduler(
            optimizer, warmup_iters, 1.0/warmup_iters,
            warmup_delay=warmup_delay_iters,
        )
    else:
        warmup_iters = 0
        warmup_lr_scheduler = None

    lr_scheduler = None

    if 'Scheduler' in config:
        scheduler_config = config['Scheduler']
        if scheduler_config['Name'] == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=scheduler_config.get('Factor', 0.1),
                patience=scheduler_config.get('Patience', 2),
                min_lr=scheduler_config.get('MinLR', 1e-5),
                cooldown=scheduler_config.get('Cooldown', 5),
            )

    # Initialize training loop variables.
    cur_iterations = 0
    summary_iters = config.get('SummaryExamples', 8192) // batch_size // args.world_size
    
    summary_prev_time = time.time()
    train_losses = [[] for _ in config['Tasks']]
    # train_losses[0] = list_train_task_losses
    num_epochs = config.get('NumEpochs', 100)
    num_iters = config.get('NumExamples', 0) // batch_size // args.world_size
    
    if primary: 
        print('training from epoch {}/{}'.format(start_epoch, num_epochs))
        print('count epoch: ', len(list_summary_epoch))
    print('previous summary_epoch {}: train_loss={} (losses={}) val_loss={} (losses={}) val={}/{} (scores={})'.format(
                        list_summary_epoch[-1],
                        list_train_loss[-1],
                        list_train_task_losses[-1],
                        list_val_loss[-1],
                        list_val_task_losses[-1],
                        list_val_score[-1],
                        best_score,
                        list_val_scores[-1],
                        # int(eval_time-summary_prev_time),
                        # int(time.time()-eval_time),
                        # optimizer.param_groups[0]['lr'],
                    ))
    
    if 'EffectiveBatchSize' in config:
        accumulate_freq = config['EffectiveBatchSize'] // batch_size // args.world_size
    else:
        accumulate_freq = 1
    model = model.to('cuda:0')
    model.train()
    for epoch in range(start_epoch, num_epochs):
        if num_iters > 0 and cur_iterations > num_iters:
            break

        if primary: print(f"Starting epoch {epoch}/{num_epochs}")

        model.train()
        optimizer.zero_grad()
        
        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        )

        for images, targets, info in train_loader_tqdm:
            cur_iterations += 1

            images = [image.to(device).float()/255 for image in images]
            gpu_targets = [
                [{k: v.to(device) for k, v in target_dict.items()} for target_dict in cur_targets]
                for cur_targets in targets
            ]

            if batch_transform:
                images, gpu_targets = batch_transform(images, gpu_targets)

            if cur_iterations == 1:
                print('input shape:', images[0].shape)

            with torch.cuda.amp.autocast(enabled=half_enabled):
                _, losses = model(images, gpu_targets)

            loss = losses.mean()
            # print('loss: ', loss)
            if loss == 0.0:
                loss = Variable(loss, requires_grad=True)

            params = [p for p in model.parameters() if p.requires_grad]
            if len(params) == 0:
                raise ValueError("Optimizer has no parameters to optimize.")
            print(f"Optimizer state dict: {optimizer.state_dict().keys()}")
            if not torch.isfinite(loss):
                print(f"Invalid loss value: {loss}")
                continue
            try:
                scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
            except Exception as e:
                print(f"Error during scaler.step: {e}")
                raise
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Loss device: {loss.device}")
            print(f"Optimizer device: {optimizer.param_groups[0]['params'][0].device}")


            if cur_iterations == 1 or cur_iterations % accumulate_freq == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loader_tqdm.set_postfix(loss=loss.item())

            for task_idx in range(len(config['Tasks'])):
                train_losses[task_idx].append(losses[task_idx].item())
            # print('train_losses[task_idx]: ',train_losses[task_idx])

            if unfreeze_iters and cur_iterations >= unfreeze_iters:
                unfreeze_iters = None
                unfreeze_hook()

            if warmup_lr_scheduler:
                warmup_lr_scheduler.step()
                if cur_iterations > warmup_delay_iters + warmup_iters + 1:
                    print('removing warmup_lr_scheduler')
                    warmup_lr_scheduler = None

            if cur_iterations % summary_iters == 0:
                train_loss = np.mean(train_losses)
                train_task_losses = [np.mean(losses) for losses in train_losses]
                # print('train_task_losses: ', train_task_losses)

                for losses in train_losses:
                    del losses[:]

                if is_distributed:
                    # Update the learning rate across all distributed nodes.
                    dist_train_loss = torch.tensor(train_loss, dtype=torch.float32, device=device)
                    torch.distributed.all_reduce(dist_train_loss, op=torch.distributed.ReduceOp.AVG)
                    if warmup_lr_scheduler is None:
                        lr_scheduler.step(dist_train_loss.item())
                else:
                    if warmup_lr_scheduler is None:
                        lr_scheduler.step(train_loss)

                # Only evaluate on the primary node (for now).
                if primary:
                    print('*** Begin evaluation ***')
                    eval_time = time.time()
                    model.eval()
                    
                    val_loss, val_task_losses, val_scores, _ = satlas.model.evaluate.evaluate(
                        config=config,
                        model=model,
                        device=device,
                        loader=val_loader,
                        half_enabled=half_enabled,
                    )
                    val_score = np.mean(val_scores)
                    # print('val_loss: {}, val_task_losses: {}'.format(val_loss, val_task_losses))
                    model.train()


                    summary_epoch += 1                   
                    print('### summary_epoch {}: train_loss={} (losses={}) val_loss={} (losses={}) val={}/{} (scores={}) elapsed={},{} lr={}'.format(
                        summary_epoch,
                        train_loss,
                        train_task_losses,
                        val_loss,
                        val_task_losses,
                        val_score,
                        best_score,
                        val_scores,
                        int(eval_time-summary_prev_time),
                        int(time.time()-eval_time),
                        optimizer.param_groups[0]['lr'],
                    ))

                    # print('list train lost before: ', list_train_loss)

                    list_epoch.append(epoch)
                    list_summary_epoch.append(summary_epoch)
                    list_train_loss.append(train_loss)
                    list_train_task_losses.append(train_task_losses)
                    list_val_loss.append(val_loss)
                    list_val_task_losses.append(val_task_losses)
                    list_val_score.append(val_score)
                    list_best_score.append(best_score)
                    list_val_scores.append(val_scores)
                    
                    summary_prev_time = time.time()

                    # print('list train lost after: ', list_train_loss)

                    # Model saving.
                    if is_distributed:
                        # Need to access underlying model in the DistributedDataParallel so keys aren't prefixed with "module.X".
                        state_dict = {
                            'model_state_dict': model.module.state_dict(),
                            'epoch': list_epoch,
                            'summary_epoch': list_summary_epoch,
                            'train_loss': list_train_loss,
                            'train_task_losses': list_train_task_losses,
                            'val_loss': list_val_loss,
                            'val_task_losses': list_val_task_losses,
                            'val_score': list_val_score,
                            'best_score': list_best_score,
                            'val_scores': list_val_scores,
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'eval_time - summary_prev_time': int(eval_time-summary_prev_time),
                            'time - eval_time': int(time.time()-eval_time),
                            'optimizer.param_groups[0][lr]': optimizer.param_groups[0]['lr']
                        }
                    else:
                        state_dict = {
                            'model_state_dict': model.state_dict(),
                            'epoch': list_epoch,
                            'summary_epoch': list_summary_epoch,
                            'train_loss': list_train_loss,
                            'train_task_losses': list_train_task_losses,
                            'val_loss': list_val_loss,
                            'val_task_losses': list_val_task_losses,
                            'val_score': list_val_score,
                            'best_score': list_best_score,
                            'val_scores': list_val_scores,
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'eval_time - summary_prev_time': int(eval_time-summary_prev_time),
                            'time - eval_time': int(time.time()-eval_time),
                            'optimizer.param_groups[0][lr]': optimizer.param_groups[0]['lr']
                        }
                    save_atomic(state_dict, save_path, 'last.pth')

                    if np.isfinite(val_score) and (best_score is None or val_score > best_score):
                        state_dict = {
                            'model_state_dict': model.state_dict(),
                            'epoch': list_epoch,
                            'summary_epoch': list_summary_epoch,
                            'train_loss': list_train_loss,
                            'train_task_losses': list_train_task_losses,
                            'val_loss': list_val_loss,
                            'val_task_losses': list_val_task_losses,
                            'val_score': list_val_score,
                            'best_score': list_best_score,
                            'val_scores': list_val_scores,
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'eval_time - summary_prev_time': int(eval_time-summary_prev_time),
                            'time - eval_time': int(time.time()-eval_time),
                            'optimizer.param_groups[0][lr]': optimizer.param_groups[0]['lr']
                        }
                        save_atomic(state_dict, save_path, 'best.pth')
                        best_score = val_score

    train_loader_tqdm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Training model.")
    parser.add_argument("--config_path", help="Configuration file path.")
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    main(args, config)
