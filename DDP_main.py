import os
import sys
import random
import numpy as np
import argparse
import torch
import fitlog
from dataloader import DiffusionLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig, AutoTokenizer
from models.modeling_roberta import RobertaForMaskedLM
from models.conditional_model import Conditional_Encoder
import Semi_AR_Diffu
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import fastNLP
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import datetime

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path",
                        default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--model_name_or_path_conditional",
                        default='princeton-nlp/sup-simcse-bert-base-uncased', type=str, required=False)
    parser.add_argument("--task_name", default='lm1b', type=str, required=False)
    parser.add_argument("--lr", default=5e-6, type=float, required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)
    parser.add_argument("--batch_size", default=48, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.1, type=float, required=False)
    parser.add_argument("--condition_lambda", default=0.2, type=float, required=False)
    parser.add_argument("--num_steps", default=2048, type=int, required=False)
    parser.add_argument("--eval_step_size", default=4, type=int, required=False)
    parser.add_argument("--dev_size", default=1e-5, type=float, required=False)
    parser.add_argument("--hybrid_lambda", default=1e-2, type=float, required=False)
    parser.add_argument("--eval_steps", default=100000, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    # parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=1000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    #parser.add_argument("--local_rank", default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    
    #python -m torch.distributed.run  --nproc_per_node 1 DDP_main.py
    args = parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=9600))

    set_seed(args)
    if args.timestep in ['none', 'token']:
        from models.modeling_bert import BertForMaskedLM
    elif args.timestep == 'layerwise':
        from models.modeling_bert_new_timestep import BertForMaskedLM
    else:
        raise NotImplementedError
    """
    if dist.get_rank() == 0:
        print('----------------------------','Set FitLog','-------------------------------')
        #git config --global user.name "Your Name"
        #git config --global user.email "youremail@yourdomain.com"
        log_dir = './logs'
        fitlog.set_log_dir(log_dir)
        fitlog.commit(__file__)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)
        #model_name_bert-base-uncased_ckpts
    """    
    save_path_m = f'./model_name_{args.model_name_or_path}_ckpts'
    save_path_con = f'./model_name_{args.model_name_or_path_conditional}_ckpts'
    
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
    word_freq = torch.load(f'./word_freq/{args.model_name_or_path}_{args.task_name}.pt')
    assert word_freq.size(0) == tokenizer.vocab_size


    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()

        # range: 0 - 1
        return wf

    def process_fn_in_collate(wf):
        return wf - wf.mean()

    word_freq = word_freq_preprocess_fn(word_freq)

    word_freq[tokenizer.pad_token_id] = 0.  # stable training

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = Semi_AR_Diffu.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = Semi_AR_Diffu.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq_lambda=args.word_freq_lambda,
        instruction_lambda=args.condition_lambda,
        device=device
    )

    if args.load_step > 0:
        ckpt_m = torch.load(os.path.join(save_path_m, f'{args.load_step}.th'))
        ckpt_c = torch.load(os.path.join(save_path_con, f'{args.load_step}.th'))
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
        ins_model.load_state_dict(ckpt_c)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    ins_model = DDP(ins_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer_m = AdamW(model.parameters(), lr=args.lr)
    optimizer_c = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler_m = torch.optim.lr_scheduler.LambdaLR(optimizer_m, lr_lambda=lambda n: n / 10000. + 1e-3 if n < 10000 else 100. / math.sqrt(n))
    warmup_scheduler_c = torch.optim.lr_scheduler.LambdaLR(optimizer_c, lr_lambda=lambda n: n / 10000. + 1e-3 if n < 10000 else 100. / math.sqrt(n))

    train_data, test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['train', 'test'])
    train_data, dev_data = train_data.train_test_split(test_size=args.dev_size).values()

    logger = fastNLP.logger
    if dist.get_rank() == 0:
        print('# of train data: {}'.format(len(train_data)))
        print('Example:')
        print(train_data[0])
        print('\n# of dev data: {}'.format(len(dev_data)))
        print('Example:')
        print(dev_data[0])
        print('\n# of test data: {}'.format(len(test_data)))
        print('Example:')
        print(test_data[0])

    def collate_fn(batch_input):
        input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
        attention_mask = [torch.tensor(d['attention_mask']) for d in batch_input]
        word_freq_logits = [process_fn_in_collate(word_freq.gather(0, torch.tensor(d['input_ids']))) for d in batch_input]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'word_freq_logits': word_freq_logits
        }


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=train_sampler)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size , collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=dev_sampler)

    cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)

    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)

    if args.timestep == 'none':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
    elif args.timestep == 'token':

        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((
                cls.repeat(bsz, 1),
                torch.full((bsz, 1), fill_value=timestep.item() + 110, device=device),
                targets,
                sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 2), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 2:-1, :]
    elif args.timestep == 'layerwise':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((
                cls.repeat(bsz, 1),
                targets,
                sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
    else:
        raise NotImplementedError

    def condition_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return ins_model(input_ids=targets, timesteps=timestep - 1, attention_mask=attention_mask)
    
    if dist.get_rank() == 0:
        if not os.path.exists(save_path_m) or not os.path.exists(save_path_con):
            os.makedirs(save_path_m, exist_ok=True)
            os.makedirs(save_path_con, exist_ok=True)
        best_dev_elbo = float('inf')

    train_loss = .0
    nan_count = 0
    loss_list = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        dev_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(tqdm(train_loader), 1):
            metrics = Semi_AR_Diffu.compute_kl_reverse_process(
                x_start=batch['input_ids'].to(device),
                t=diffusion_instance.sample_t(),
                denoise_fn=denoise_fn,
                diffusion=diffusion_instance,
                instruction=batch['input_ids'].to(device),
                instruction_fn=condition_fn,
                target_mask=batch['attention_mask'].to(device),
                hybrid_lambda=args.hybrid_lambda,
                predict_x0=args.predict_x0,
                word_freq_logits=batch['word_freq_logits'].to(device)
            )

            loss = metrics['loss'] / args.batch_size
            dist.all_gather(loss_list, loss)
            if torch.stack(loss_list).isnan().any():
                nan_count += 1
                logger.warning(f'NaN encountered {nan_count} times')
                continue
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer_m.step()
            optimizer_c.step()
            model.zero_grad()
            ins_model.zero_grad()
            optimizer_m.zero_grad()
            optimizer_c.zero_grad()
            warmup_scheduler_m.step()
            warmup_scheduler_c.step()
            
            if dist.get_rank() == 0:
                if i % args.logging_steps == args.logging_steps - 1:
                    logger.info(f'Loss at step {i} is {train_loss / args.logging_steps}')
                    #fitlog.add_loss(train_loss / args.logging_steps, name='train_loss', step=i)

                    torch.save(model.state_dict(), f'./{save_path_m}/{i}.th')
                    torch.save(ins_model.state_dict(), f'./{save_path_con}/{i}.th')
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
                        batch_dev_metrics = Semi_AR_Diffu.compute_kl_reverse_process(
                            x_start=batch['input_ids'].to(device),
                            t=diffusion_instance.sample_t().to(device),
                            denoise_fn=denoise_fn,
                            diffusion=diffusion_instance,
                            instruction=batch['input_ids'].to(device),
                            instruction_fn=condition_fn,
                            target_mask=batch['attention_mask'].to(device),
                            hybrid_lambda=args.hybrid_lambda,
                            predict_x0=args.predict_x0,
                            word_freq_logits=batch['word_freq_logits'].to(device)
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
                            dev_metrics[name] /= len(dev_data)
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
                                'optimizer_m': optimizer_c.state_dict(),
                                'warmup_scheduler': warmup_scheduler_c.state_dict(),
                            }, f'./{save_path_con}/best({i}).th')
                    model.train()

