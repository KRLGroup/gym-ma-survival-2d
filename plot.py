from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import tensorboard as tb
import pandas as pd

import experiments


def extract_run(df, run, tags):
    run_df = pd.DataFrame(columns=['step'])
    for i, tag in enumerate(tags):
        mask = (df['run'] == run) & (df['tag'] == tag)
        tag_df = df[mask][['step', 'value']].rename(columns={'value': tag})
        run_df = run_df.merge(tag_df, on='step', how='outer')
    return run_df

def plot_tags(exp_df, tags, tag2label, title, subplots, smoothing=0.6):
    runs = exp_df['run'].unique()
    assert len(runs) == 1
    run = runs[0]
    labels = list(tag2label.values())
    run_df = extract_run(exp_df, run, tags).rename(columns=tag2label)
    smoothed = run_df.ewm(alpha=(1 - smoothing)).mean()
    fig, axs = plt.subplots(len(subplots))
    fig.suptitle(title)
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for ax, (subtitle, subplot) in zip(axs, subplots.items()):
        for label, column in subplot.items():
            color = next(colors)
            ax.plot(run_df['step'], run_df[column], alpha=0.2, color=color)
            ax.plot(
                smoothed['step'], smoothed[column], label=label, color=color
            )
        #ax.set_title(subtitle)
        ax.set(xlabel='steps', ylabel=subtitle)
        ax.legend()
    for ax in axs:
        ax.label_outer()
#     run_df = run_df.rename(columns={
#         label: '_raw_' + label for label in labels
#     })
#     df = run_df.merge(smoothed, on='step', how='outer')
    # "subplots": [["r1", "r2"], ["k1", "k2"], ["#heals", "#boxes"]]
    #kwargs = dict(x='step', y=labels, colormap='tab10', subplots=[('r1', 'r2'), ('k1', 'k2'), ("#heals", "#boxes")], **args)
    #ax = smoothed.plot(**kwargs)
    #run_df.plot(ax=ax, alpha=0.4, legend=False, **kwargs)
    #plt.show()



# Script stuff

def main(exp_dirpath, dpi):
    print(f'Plotting the only run of experiment at {exp_dirpath}.')
    exp = experiments.get_experiment(exp_dirpath)
    remote_exp = tb.data.experimental.ExperimentFromDev(exp['tb_dev_id'])
    exp_df = remote_exp.get_scalars()
    plot_tags(exp_df, **exp['plot'])
    plot_fpath = f'{exp["name"]}.png'
    print(f'Saving plot at {plot_fpath}.')
    plt.savefig(plot_fpath, dpi=dpi)

argparse_desc = \
"""Produce plots from tensorboard logs uploaded to tensorboard/dev.
"""

argparse_args = [
    (['exp_dirpath'], {
        'metavar': 'DIR',
        'type': str,
        'help': 'The experiment directory.',
    }),
    (['--dpi'], {
        'dest': 'dpi',
        'metavar': 'DPI',
        'default': 256,
        'help': 'The DPI to use when saving the plot image.',
    }),
#     (['-t', '--tags'], {
#         'required': True,
#         'metavar': 'T',
#         'type': str,
#         'nargs': '+',
#         'help': 'Tags to extract from the tensorboard runs.',
#     }),
#     (['-o', '--output'], {
#         'dest': 'out_fpath',
#         'metavar': 'OUTPATH',
#         'default': None,
#         'help': 'The path to save the concatenated dataframe to.',
#     }),
]

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=argparse_desc)
    for args, kwargs in argparse_args:
        argparser.add_argument(*args, **kwargs)
    cli_args = argparser.parse_args()
    #print(vars(cli_args))
    main(**vars(cli_args))
