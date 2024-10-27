import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.9'
import jax
import optax
import jax.numpy as jnp
from jax import random
from data.cifar10 import cifar10
from data.cifar100 import cifar100
from data.mnist import mnist
from functools import partial
from jax.tree_util import Partial
from tqdm import tqdm
import argparse
import haiku as hk
from logger import Logger
from metaoptim import inner_step, loss_func, acc, grad_implicit_func
from Models.ConvNet import ConvNet
from torch.utils.data import DataLoader


def _one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def construct_random_data(
        key, 
        ipc=1, num_classes=10, 
        img_size=32, num_channels=3, 
        load_checkpoint=False, ckpt_path=""
    ):
    if not load_checkpoint:
        random_images = random.uniform(key, [num_classes * ipc, img_size, img_size, num_channels])
        labels = jnp.eye(num_classes)
        labels = jnp.tile(labels, (ipc, 1))
        return random_images, labels
    else:
        return jnp.load(ckpt_path + '_img.npy'), jnp.load(ckpt_path + '_label.npy')

@partial(jax.jit, static_argnums=(1, 2, 3))
def init(key, model, img_size, num_channels): 
    return model.init(key, jnp.ones([1, img_size, img_size, num_channels]), True)


def data_stream(train_dataset, batch_size, num_classes, shuffle, drop_last=False):
    while True:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        for train_imgs, train_labels in train_dataloader:
            yield train_imgs.numpy(), _one_hot(train_labels.numpy(), num_classes)

parser = argparse.ArgumentParser()
parser.add_argument('--inner_step_size', type=float, default=1e-2)
parser.add_argument('--num_inner_steps', type=int, default=100)
parser.add_argument('--eval_num_inner_steps', type=int, default=100)
parser.add_argument('--truncated_steps', type=int, default=100)
parser.add_argument('--meta_step_size', type=float, default=1e-2)
parser.add_argument('--principle', type=str, default="neumann")
parser.add_argument('--alpha', type=float, default=1.) # parameter for neumann 0.01
parser.add_argument('--inner_weight_decay', type=float, default=1e-4)
parser.add_argument('--num_max_steps', type=int, default=3)  # for ihvp 20
parser.add_argument('--num_steps', type=int, default=10000)
parser.add_argument('--model', type=str, default="ConvNet")
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_clip', action='store_true')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_expname', type=str, default='implicit_ipc1_cifar_inner100')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval_step', type=int, default=100)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--ckpt_path', type=str, default="")
parser.add_argument('--inner_batch_size', type=int, default=-1)
parser.add_argument('--ipc', type=int, default=1)
args = parser.parse_args()

logger = Logger(
    'seed={}'.format(args.seed),
    root_dir='cifar_data_condensation_3', with_timestamp=True)

logger.add_params(vars(args))
if args.wandb: logger.wandb_init(project="BILEVEL_DC", name=args.wandb_expname)

if __name__ == "__main__":
    rng = random.PRNGKey(args.seed)
    if args.dataset.lower() == 'cifar10':
        train_dataset, test_dataset = cifar10()
        num_classes = 10
        img_size = 32
        num_channels = 3
    elif args.dataset.lower() == 'cifar100':
        train_dataset, test_dataset = cifar100()
        num_classes = 100
        img_size = 32
        num_channels = 3
    elif args.dataset.lower() == 'mnist':
        train_dataset, test_dataset = mnist()
        num_classes = 10
        img_size = 28
        num_channels = 1
    else:
        raise ValueError("Invalid dataset")

    num_train = len(train_dataset)
    num_test = len(test_dataset)
    rng, key = random.split(rng)
    inner_batch_size = args.inner_batch_size if args.inner_batch_size != -1 else args.ipc * num_classes
    assert inner_batch_size <= args.ipc * num_classes
    condensated_images, condensated_labels = construct_random_data(
        key, 
        args.ipc, 
        num_classes, img_size, num_channels, 
        args.load_checkpoint, args.ckpt_path
    )

    batches = data_stream(train_dataset, args.batch_size, num_classes, shuffle=True, drop_last=False)

    @hk.transform
    def network(img, is_training):
        net = ConvNet(num_classes=num_classes)
        return net(img, is_training)
    model = hk.without_apply_rng(network)
    
    inner_optimizer = optax.sgd(
        optax.cosine_decay_schedule(
            init_value=args.inner_step_size, decay_steps=args.num_inner_steps, alpha=0.0
        ),
        momentum=0.9
    )

    eval_optimizer = optax.sgd(
        optax.cosine_decay_schedule(
            init_value=args.inner_step_size, decay_steps=args.eval_num_inner_steps, alpha=0.0
        ),
        momentum=0.9
    )

    meta_optimizer = optax.adam(
        optax.linear_schedule(
            init_value=args.meta_step_size, end_value=0,
            transition_steps=args.num_steps,
            transition_begin=0
        )
    )
    meta_opt_state = meta_optimizer.init(condensated_images)
    total_steps = 0
    pbar = tqdm(total=args.num_steps)
    rank_size = jax.local_device_count()

    while total_steps < args.num_steps:
        # evaluation
        if total_steps % args.eval_step == 0:
            rng, key = random.split(rng)
            eval_params = init(key, model, img_size, num_channels)
            eval_inner_opt_state = eval_optimizer.init(eval_params)
            for i in range(args.eval_num_inner_steps):
                rng, key = random.split(rng)
                random_indexes = jax.random.permutation(rng, len(condensated_images))[:inner_batch_size]
                mini_params = condensated_images[random_indexes]
                mini_labels = condensated_labels[random_indexes]
                eval_params, eval_inner_opt_state, eval_loss, eval_logits = inner_step(
                    eval_params,
                    eval_inner_opt_state,                    
                    mini_params,
                    mini_labels,
                    Partial(model.apply),
                    Partial(eval_optimizer.update),
                    args.inner_weight_decay
                )
            meta_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            meta_losses = []
            meta_acces = []
            test_acces = []
            for train_imgs, train_labels in meta_dataloader:
                train_imgs, train_labels = train_imgs.numpy(), _one_hot(train_labels.numpy(), num_classes)
                batch_size = train_imgs.shape[0]
                meta_loss_query, _ = loss_func(
                    eval_params, 
                    train_imgs,
                    train_labels,
                    Partial(model.apply),
                )
                meta_acc_query = acc(
                    eval_params, 
                    train_imgs,
                    train_labels,
                    Partial(model.apply),
                )
                meta_losses.append(meta_loss_query * batch_size)
                meta_acces.append(meta_acc_query * batch_size)

            for test_imgs, test_labels in test_dataloader:
                test_imgs, test_labels = test_imgs.numpy(), _one_hot(test_labels.numpy(), num_classes)
                batch_size = test_imgs.shape[0]
                test_acc = acc(
                    eval_params, 
                    test_imgs,
                    test_labels,
                    Partial(model.apply)
                )
                test_acces.append(test_acc * batch_size)

            jnp.save('checkpoints/' + args.dataset + "/" + args.wandb_expname + '_img.npy', condensated_images)
            jnp.save('checkpoints/' + args.dataset + "/" + args.wandb_expname + '_label.npy', condensated_labels)

            meta_loss = jnp.sum(jnp.array(meta_losses)) / num_train
            meta_acc = jnp.sum(jnp.array(meta_acces)) / num_train
            test_acc = jnp.sum(jnp.array(test_acces)) / num_test

            logger.add_metric('meta_loss', float(meta_loss))
            logger.add_metric('meta_acc', float(meta_acc))
            logger.add_metric('test_acc', float(test_acc))
            logger.commit(epoch=0, step=total_steps)

        with logger.measure_time('step'):
            batch = next(batches)
            rng, key = random.split(rng)
            inner_params = init(key, model, img_size, num_channels)
            inner_opt_state = inner_optimizer.init(inner_params)
            for i in range(args.num_inner_steps - args.truncated_steps):
                rng, key = random.split(rng)
                random_indexes = jax.random.permutation(rng, len(condensated_images))[:inner_batch_size]
                mini_params = condensated_images[random_indexes]
                mini_labels = condensated_labels[random_indexes]
                inner_params, inner_opt_state, _, _ = inner_step(
                    inner_params,
                    inner_opt_state,                    
                    mini_params,
                    mini_labels,
                    Partial(model.apply),
                    Partial(inner_optimizer.update),
                    args.inner_weight_decay
                )

            g_approx = grad_implicit_func(
                key,
                inner_params,
                condensated_images, condensated_labels,
                batch,
                Partial(model.apply),
                args.alpha, args.num_max_steps,
                args.principle,
                args.inner_weight_decay,
                inner_batch_size
            )
        meta_updates, meta_opt_state = meta_optimizer.update(g_approx, meta_opt_state)
        condensated_images = optax.apply_updates(condensated_images, meta_updates)
        if not args.no_clip: condensated_images = condensated_images.clip(0., 1.)

        total_steps += 1
        pbar.update(1)
    pbar.close()
    jnp.save('checkpoints/' + args.dataset + "/" + args.wandb_expname + '_img.npy', condensated_images)
    jnp.save('checkpoints/' + args.dataset + "/" + args.wandb_expname + '_label.npy', condensated_labels)
