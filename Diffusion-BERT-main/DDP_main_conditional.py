import functools
import os
import sys
import random
import numpy as np
import argparse
import torch
import fitlog
from models.conditional_model import Conditional_Encoder
from dataloader import QQPLoader, QTLoader, XSumLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_bert import BertForMaskedLM
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_condition
from torch.optim import AdamW
import fastNLP
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200, type=int, required=False)
    parser.add_argument("--model_name_or_path",
                        default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--model_name_or_path_conditional",
                        default='princeton-nlp/sup-simcse-bert-base-uncased', type=str, required=False)
    #parser.add_argument("--model_name_or_path", default='roberta-base', type=str, required=False)
    parser.add_argument("--task_name", default='XSum', type=str, required=False)
    parser.add_argument("--lr", default=2e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=18, type=int, required=False)
    parser.add_argument("--dev_size", default=1e-1, type=float, required=False)
    parser.add_argument("--word_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=2000, type=int, required=False)
    parser.add_argument("--eval_step_size", default=80, type=int, required=False)
    parser.add_argument("--accumulation_steps", default=4, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, required=False)
    parser.add_argument("--eval_steps", default=2000, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=200, type=int, required=False)
    parser.add_argument("--save_steps", default=2000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    #python -m torch.distributed.run  --nproc_per_node 1 DDP_main_conditional.py
    args = parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=9600))
    set_seed(args)
    """
    if dist.get_rank() == 0:
        log_dir = './logs'
        fitlog.set_log_dir(log_dir)
        fitlog.commit(__file__)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)
    """
    Dataloaders = {
        'qqp': QQPLoader,
        'QT': QTLoader,
        'XSum':XSumLoader
    }

    Loader = Dataloaders[args.task_name]

    save_path_m = f'./model_name_{args.model_name_or_path}_ckpts'
    save_path_i = f'./model_name_{args.model_name_or_path_conditional}_ckpts'
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    else:
        raise NotImplementedError

    tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
    word_freq = torch.zeros(tokenizer.vocab_size)
    assert word_freq.size(0) == tokenizer.vocab_size


    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()

        # range: 0 - 1
        return wf


    word_freq = word_freq_preprocess_fn(word_freq)

    # word_freq[tokenizer.pad_token_id] = 0.  # stable training

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_condition.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )
    
    if args.load_step > 0:
        ckpt_m = torch.load(os.path.join(save_path_m, f'{args.load_step}.th'))
        ckpt_i = torch.load(os.path.join(save_path_i, f'{args.load_step}.th'))
    cfg = cfg_cls.from_pretrained(args.model_name_or_path)
    cfg.overall_timestep = diffusion_instance.num_steps

    if args.from_scratch:
        model = model_cls(cfg).to(device).to(device)
        ins_model = Conditional_Encoder().to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
        ins_model = Conditional_Encoder().to(device)
    else:
        model = model_cls(cfg).to(device)
        model.load_state_dict(ckpt_m)
        ins_model = Conditional_Encoder().to(device)
        ins_model.load_state_dict(ckpt_i)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    ins_model = DDP(ins_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    ckptm = torch.load('./model_name_bert-base-uncased_ckpts/269999.th')
    ckptc = torch.load('./model_name_princeton-nlp/sup-simcse-bert-base-uncased_ckpts/39999.th')
    model.load_state_dict(ckptm)
    ins_model.load_state_dict(ckptc)
    
    optimizer_m = AdamW(model.parameters(), lr=args.lr)
    optimizer_i = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler_m = torch.optim.lr_scheduler.LinearLR(optimizer_m, total_iters=10000)
    warmup_scheduler_i = torch.optim.lr_scheduler.LinearLR(optimizer_i, total_iters=10000)
    
    train_data, test_data = Loader(tokenizer=tokenizer).my_load(splits=['train', 'test'])
    test_data, dev_data = test_data.train_test_split(test_size=args.dev_size).values()

    if dist.get_rank() == 0:
        logger = fastNLP.logger
        print('# of train data: {}'.format(len(train_data)))
        print('Example:')
        print(train_data[0])
        print('\n# of dev data: {}'.format(len(dev_data)))
        print('Example:')
        print(dev_data[0])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
                                               num_workers=4, pin_memory=True, sampler=train_sampler)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
                                             num_workers=4, pin_memory=True, sampler=dev_sampler)

    
    cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)

    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)
    
    if args.load_step > 0:
        optimizer_m.load_state_dict(ckpt_m['optimizer'])
        warmup_scheduler_m.load_state_dict(ckpt_m['warmup_scheduler'])
        optimizer_i.load_state_dict(ckpt_i['optimizer'])
        warmup_scheduler_i.load_state_dict(ckpt_i['warmup_scheduler'])
    model.train()

    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        # input_ids 'I am from China. I am Chinese. [PAD][PAD]'
        # corrupted_input_ids  '[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        # target mask  '0 0 0 0 0 1 1 1 1 1 0 0'
        # new_input_ids  'I am from China. [MASK] [MASK] [MASK] [MASK] [PAD] [PAD]'

        # input_ids 'I am from China. I am Chinese. [PAD][PAD]'
        # corrupted_input_ids  '[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        # target mask  '0 0 0 0 0 1 1 1 1 1 1 1'
        # new_input_ids  'I am from China. [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        return model(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
        )['logits']

    def instruction_fn(targets, timestep, attention_mask=None):
        #assert len(targets.size()) == 2  # bsz * seqlen
        #bsz = targets.size(0)
        #targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        #attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return ins_model(input_ids=targets, timesteps=timestep - 1, attention_mask=attention_mask)
    
    if dist.get_rank() == 0:
        if not os.path.exists(save_path_m):
            os.makedirs(save_path_m, exist_ok=True)
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i, exist_ok=True)
        best_dev_elbo = float('inf')

    train_loss = .0
    nan_count = 0
    loss_list = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
    i = -1
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        dev_loader.sampler.set_epoch(epoch)
        for batch in tqdm(train_loader):
            i += 1
            for k, v in batch.items():
                batch[k] = v.to(device)
            t = diffusion_instance.sample_t()
            metrics = diffusion_condition.compute_kl_reverse_process(
                batch['input_ids'],
                t.to(device),
                instruction=batch['instruction'],
                instruction_fn=instruction_fn,
                denoise_fn=functools.partial(
                    denoise_fn,
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target_mask=batch['target_mask']
                ),
                diffusion=diffusion_instance,
                target_mask=batch['target_mask'],
                hybrid_lambda=args.hybrid_lambda,
                predict_x0=args.predict_x0,
                word_freq_logits=torch.zeros_like(batch['input_ids'])
            )

            loss = metrics['loss']
            dist.all_gather(loss_list, loss)
            if torch.stack(loss_list).isnan().any():
                nan_count += 1
                if dist.get_rank() == 0:
                    logger.warning(f'NaN encountered {nan_count} times')
                continue
            train_loss += loss.item()
            loss = loss / args.accumulation_steps
            loss.backward()
            # diffusion_instance.update_loss(t.numpy(), loss.item())
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            if i % args.accumulation_steps == args.accumulation_steps - 1:
                optimizer_m.step()
                optimizer_i.step()
                model.zero_grad()
                ins_model.zero_grad()
                optimizer_m.zero_grad()
                optimizer_i.zero_grad()
                warmup_scheduler_m.step()
                warmup_scheduler_i.step()

            if dist.get_rank() == 0:
                if i % args.logging_steps == args.logging_steps - 1:
                    logger.info(f'Loss at step {i} is {train_loss / args.logging_steps}')
                    #fitlog.add_loss(train_loss / args.logging_steps, name='train_loss', step=i)

                train_loss = .0
            if i % args.eval_steps == args.eval_steps - 1:
                nan_count_in_dev = 0
                model.eval()
                ins_model.eval()
                dev_metrics = {
                    'elbo': .0,
                    'elbo_in_bits_per_dim': .0,
                    # 'likelihood': .0,
                    # 'prior': .0,
                }
                with torch.no_grad():
                    for dev_batch in dev_loader:
                        for k, v in dev_batch.items():
                            dev_batch[k] = v.to(device)
                        batch_dev_metrics = diffusion_condition.discrete_diffusion_elbo(
                            dev_batch['input_ids'],
                            instruction=dev_batch['instruction'],
                            instruction_fn=instruction_fn,
                            denoise_fn=functools.partial(
                                denoise_fn,
                                input_ids=dev_batch['input_ids'],
                                attention_mask=dev_batch['attention_mask'],
                                target_mask=dev_batch['target_mask']
                            ),
                            diffusion=diffusion_instance,
                            target_mask=dev_batch['target_mask'],
                            normalize_without_padding=True,
                            eval_step_size=args.eval_step_size,
                            word_freq_logits=torch.zeros_like(dev_batch['input_ids'])
                        )
                        if dist.get_rank() == 0:
                            m = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
                            for name in dev_metrics.keys():
                                dist.gather(batch_dev_metrics[name].squeeze(), m)
                                temp = sum(m)
                                if not torch.isnan(temp):
                                    dev_metrics[name] += temp
                                else:
                                    nan_count_in_dev += 1
                                    logger.warning(f'NaN encountered {nan_count_in_dev} times in dev')
                        else:
                            for name in dev_metrics.keys():
                                dist.gather(batch_dev_metrics[name].squeeze())
                    if dist.get_rank() == 0:
                        for name in dev_metrics.keys():
                            dev_metrics[name] /= (len(dev_data) - nan_count_in_dev * 2 * args.batch_size)
                            #fitlog.add_metric(dev_metrics[name], name=name, step=i)
                        if dev_metrics['elbo_in_bits_per_dim'] <= best_dev_elbo:
                            best_dev_elbo = dev_metrics['elbo_in_bits_per_dim']
                            #fitlog.add_best_metric(dev_metrics['elbo_in_bits_per_dim'], name='dev_elbo_in_bits_per_dim')
                            torch.save({
                                'model': model.state_dict(),
                                'optimizer_m': optimizer_m.state_dict(),
                                'warmup_scheduler': warmup_scheduler_m.state_dict(),
                            }, f'./{save_path_m}/best({i}).th')
                            torch.save({
                                'model': ins_model.state_dict(),
                                'optimizer': optimizer_i.state_dict(),
                                'warmup_scheduler': warmup_scheduler_i.state_dict(),
                            }, f'./{save_path_i}/best({i}).th')
                model.train()
            #
            # if i % args.save_steps == args.save_steps - 1:
            #     torch.save({
            #         'model': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'warmup_scheduler': warmup_scheduler.state_dict(),
            #     }, f'{save_path}/{i}.th')
