import argparse
import os
from tqdm import tqdm
import pprint as pp
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value
from neural_combinatorial_rl import NeuralCombOptNet
import datasets
from constants import constants
import nets
from datasets import matches_sum, found_matches_sum, found_matches_portion


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

# Data
parser.add_argument('--batch_size', default=128, help='')
# Network
parser.add_argument('--embedding_dim', default=0, help='Dimension of input embedding')
parser.add_argument('--hidden_dim', default=128, help='Dimension of hidden layers in Enc/Dec')
parser.add_argument('--n_process_blocks', default=3, help='Number of process block iters to run in the Critic network')
parser.add_argument('--n_glimpses', default=2, help='No. of glimpses to use in the pointer network')
parser.add_argument('--use_tanh', type=str2bool, default=True)
parser.add_argument('--tanh_exploration', default=10,
                    help='Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
parser.add_argument('--dropout', default=0., help='')
parser.add_argument('--net_type', type=str, default='pointer_net')
parser.add_argument('--encoder_bi_directional', type=str2bool, default=False)
parser.add_argument('--encoder_num_layers', type=int, default=1)
# Training
parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
parser.add_argument('--actor_lr_decay_step', default=5000, help='')
parser.add_argument('--critic_lr_decay_step', default=5000, help='')
parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
parser.add_argument('--reward_scale', default=2, type=float, help='')
parser.add_argument('--is_train', type=str2bool, default=True, help='')
parser.add_argument('--n_epochs', type=int, default=1, help='')
parser.add_argument('--random_seed', default=24601, help='')
parser.add_argument('--max_grad_norm', default=2.0, help='Gradient clipping')
parser.add_argument('--use_cuda', type=str2bool, default=True, help='')
parser.add_argument('--critic_beta', type=float, default=0.9, help='Exp mvg average decay')
parser.add_argument('--pad_value', type=float, default=-1)
parser.add_argument('--repeat_batch', type=int, default=1)
# Misc
parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--output_dir', type=str, default='training_data')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--disable_tensorboard', type=str2bool, default=False)

args = vars(parser.parse_args())

# Pretty print the run args
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))

# Optionally configure tensorboard
if not args['disable_tensorboard']:
    configure(os.path.join(args['log_dir'], args['run_name']))

# Task specific configuration - generate dataset if needed
training_dataset = datasets.Dataset("train", args['repeat_batch'])
val_dataset = datasets.Dataset("test", args['repeat_batch'])

# Load the model parameters from a saved state
if args['load_path'] != '':
    print('  [*] Loading model from {}'.format(args['load_path']))

    model = torch.load(os.path.join(os.getcwd(), args['load_path']))

    model.actor_net.decoder.max_length = constants['window_size'] + 1
    model.is_train = args['is_train']
else:
    # Instantiate the Neural Combinatorial Opt with RL module
    model = NeuralCombOptNet(
        constants['event_size'],
        int(args['embedding_dim']),
        int(args['hidden_dim']),
        constants['window_size'] + 1,  # decoder len
        int(args['n_glimpses']),
        float(args['tanh_exploration']),
        args['use_tanh'],
        args['is_train'],
        args['use_cuda'],
        args['encoder_bi_directional'],
        args['encoder_num_layers']
    ) if args['net_type'] == 'pointer_net' else nets.NeuralCombOptLinearNet_OLD(args['batch_size'], args['use_cuda'])

save_dir = os.path.join(os.getcwd(), args['output_dir'], args['run_name'])

if args['net_type'] != 'pointer_net':
    args['pad_value'] = None

try:
    os.makedirs(save_dir)
except:
    pass

# critic_mse = torch.nn.MSELoss()
# critic_optim = optim.Adam(model.critic_net.parameters(), lr=float(args['critic_net_lr']))
actor_optim = optim.Adam(model.actor_net.parameters(), lr=float(args['actor_net_lr']))

actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
                                           range(int(args['actor_lr_decay_step']),
                                                 int(args['actor_lr_decay_step']) * 1000,
                                                 int(args['actor_lr_decay_step'])),
                                           gamma=float(args['actor_lr_decay_rate']))

# critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
#        range(int(args['critic_lr_decay_step']), int(args['critic_lr_decay_step']) * 1000,
#            int(args['critic_lr_decay_step'])), gamma=float(args['critic_lr_decay_rate']))

training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']))

validation_dataloader = DataLoader(val_dataset, batch_size=1)

critic_exp_mvg_avg = torch.zeros(1)
beta = args['critic_beta']

if args['use_cuda']:
    model = model.cuda()
    # critic_mse = critic_mse.cuda()
    critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

train_step = 0
val_step = 0

if not args['is_train']:
    args['n_epochs'] = 1

epoch = int(args['epoch_start'])


def step(model, batch):
    x, m = batch
    m = m.squeeze(1)
    if args['pad_value'] is not None:
        beginning_padding = torch.full(size=(x.size(0), x.size(1), 1), fill_value=args['pad_value'], dtype=x.dtype)
        x = torch.cat((beginning_padding, x), dim=2)
    if args['use_cuda']:
        x = x.cuda()
        m = m.cuda()
    probs, actions, actions_idxs = model(x)
    R, batch_found_matches_sum, batch_matches_sum, chosen_events_portion = datasets.get_rewards(m, torch.stack(
        actions_idxs).T)
    if args['use_cuda']:
        R = R.cuda()
    return R, probs, actions, actions_idxs, x, batch_found_matches_sum, batch_matches_sum, chosen_events_portion


def step_OLD(model, batch):
    x, m = batch
    if args['use_cuda']:
        x = x.cuda()
    chosen_events_np, log_prob = model(x)
    R = datasets.get_rewards_OLD(m, chosen_events_np)
    if args['use_cuda']:
        R = R.cuda()
    return R, log_prob, chosen_events_np


repeat = 0

for i in range(epoch, epoch + args['n_epochs']):

    if args['is_train']:

        model.train()

        # sample_batch is [batch_size, event_size, window_size]
        # for batch_id, batch in enumerate(training_dataloader if args['repeat_batch'] != 1 else tqdm(training_dataloader)):
        for batch_id, batch in enumerate(datasets.DataloaderOLD("train", args['batch_size'])):
            # for _ in range(args['repeat_batch']):

            # R, probs, actions, actions_idxs, x, batch_found_matches_sum, batch_matches_sum, chosen_events_num = \
            #     step(model, batch)

            R, probs, chosen_events = step_OLD(model, batch)

            if args['net_type'] == 'pointer_net':
                log_probs = torch.zeros_like(probs[0])

                finished_batches_mask = torch.ones_like(probs[0]).int()

                if args['use_cuda']:
                    log_probs = log_probs.cuda()
                    finished_batches_mask = finished_batches_mask.cuda()

                for prob, idxs in zip(probs, actions_idxs):
                    # compute the sum of the log probs
                    # for each tour in the batch
                    log_prob = torch.log(prob)
                    # zero the log_prob where previous(!) batch action is 0 (=finished)
                    log_prob = log_prob * finished_batches_mask
                    finished_batches_mask = finished_batches_mask * idxs.bool().int()

                    log_probs += log_prob
            else:  # net is fully connected
                log_probs = probs

            nll = -log_probs

            # guard against nan
            nll[(nll != nll).detach()] = 0.
            # clamp any -inf's to 0 to throw away this tour
            log_probs[(log_probs < -1000).detach()] = 0.

            if batch_id == 0:
                critic_exp_mvg_avg = R.mean()
            else:
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

            advantage = R - critic_exp_mvg_avg

            # multiply each time step by the advantage
            reinforce = advantage * log_probs
            actor_loss = reinforce.mean()

            actor_optim.zero_grad()

            actor_loss.backward()

            # clip gradient norms
            torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(), float(args['max_grad_norm']), norm_type=2)

            actor_optim.step()
            actor_scheduler.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            # critic_scheduler.step()

            # R = R.detach()
            # critic_loss = critic_mse(v.squeeze(1), R)
            # critic_optim.zero_grad()
            # critic_loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(),
            #        float(args['max_grad_norm']), norm_type=2)

            # critic_optim.step()

            train_step += 1

            # if not args['disable_tensorboard']:
            #     log_value('avg_reward', R.mean().item(), train_step)
            #     log_value('actor_loss', actor_loss.item(), train_step)
            #     # log_value('critic_loss', critic_loss.item(), train_step)
            #     log_value('critic_exp_mvg_avg', critic_exp_mvg_avg.item(), train_step)
            #     log_value('nll', nll.mean().item(), train_step)
            #found_matches_portion = batch_found_matches_sum / batch_matches_sum if batch_matches_sum != 0 else 1
            if train_step % args['log_step'] == 0:
                # print('\nepoch: {}, train_batch_id: {}, avg_reward: {}, found matches: {}/{}, found matches '
                #       'portion: {}, chosen events portion: {}/{}'.format(i, batch_id, R.mean().item(),
                #                                                          batch_found_matches_sum, batch_matches_sum,
                #                                                          found_matches_portion, chosen_events_num,
                #                                                          constants['window_size']))
                print("Epoch " + str(epoch) + ": Processed " + " out of "
                      + str(constants['train_size']) + " sampled reward of " + str(R.mean().item()) + "\n" +
                      "and chosen events " + str(chosen_events[0]) + "\n" + "found matches portion: "
                      + str(found_matches_portion))
                # example_output = set()
                # for action_idx in actions_idxs:
                #     action_idx = action_idx[0].item()
                #     example_output.add(action_idx)
                # output = torch.zeros(constants['window_size']).int()
                # output[[i - 1 for i in example_output if i != 0]] = 1
                # print('Example train output: {}'.format(output.tolist()))

    print('\n~Validating~\n')

    example_input = []
    example_output = []
    avg_reward = []

    # put in test mode!
    model.eval()

    for batch_id, batch in enumerate(tqdm(validation_dataloader)):
        R, probs, actions, actions_idxs, x, found_matches_portion, chosen_events_num = step(model, batch)

        avg_reward.append(R[0].item())
        val_step += 1.

        if not args['disable_tensorboard']:
            log_value('val_avg_reward', R[0].item(), int(val_step))

        if val_step % args['log_step'] == 0:
            example_output = set()
            for action_idx in actions_idxs:
                action_idx = action_idx[0].item()
                example_output.add(action_idx)
            output = torch.zeros(constants['window_size']).int()
            output[[i - 1 for i in example_output if i != 0]] = 1
            print('Example train output: {}'.format(output.tolist()))
            print('Example test reward: {}'.format(R[0].item()))

    print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
    print('Validation overall reward var: {}'.format(np.var(avg_reward)))

    if args['is_train']:
        print('Saving model...')
        torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))
