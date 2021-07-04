import os

import numpy as np


def read_file(folder_path):
    with open(os.path.join(folder_path, 'finish.txt'), 'r') as f:
        lines = f.readlines()
    n = [list(map(float, l.strip().split()[1:3])) for l in lines]
    return np.asarray(n)


folder = '/samsung_hdd/Files/AI/TNO/remote_folders/pixel_nerf_eval/pixel_nerf_eval_output/sncar'
# folder = '/samsung_hdd/Files/AI/TNO/remote_folders/pixel_nerf_eval/pixel_nerf_eval_output/snship'
nm = read_file(os.path.join(folder, 'no_mirror'))
sl = read_file(os.path.join(folder, 'sym_loss'))

print(f"No mirror: {np.mean(nm, axis=0)} over {len(nm)} objects")
print(f"Sym loss: {np.mean(sl, axis=0)} over {len(sl)} objects")
