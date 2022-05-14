import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import ants
import h5py
import nibabel as nib

import brainsss


def main(args):
    logfile = args['logfile']
    directory = args['directory']
    carrier_path = args['carrier']
    passenger_path = args['passenger']
    step_size = args['step_size']
    width = 120
    printlog = getattr(brainsss.Printlog(logfile=logfile), 'print_to_log')

    moco_dir = os.path.join(directory, 'moco')
    if not os.path.exists(moco_dir):
        os.makedirs(moco_dir)

    carrier = nib.load(carrier_path)
    passenger = nib.load(passenger_path)
    fixed_path = carrier_path.split('.')[0] + '_mean.nii'
    fixed = np.asarray(nib.load(fixed_path).get_data(), dtype='float32')
    fixed = ants.from_numpy(fixed)

    assert carrier.header.get_data_shape() == passenger.header.get_data_shape()
    brain_shape = carrier.header.get_data_shape()

    carrier_h5 = carrier_path.split('.')[0] + '_moco.h5'
    make_empty_h5(carrier_h5, brain_shape)

    passenger_h5 = passenger_path.split('.')[0] + '_moco.h5'
    make_empty_h5(passenger_h5, brain_shape)

    total_volumes = brain_shape[-1]
    n_steps = np.ceil(total_volumes / step_size)

    steps = list(range(0, total_volumes, step_size))
    # add the last few volumes that are not divisible by stepsize
    if total_volumes > steps[-1]:
        steps.append(total_volumes)

    printlog(F"{'   STARTING MOCO   ':-^{width}}")
    for i in range(n_steps):
        carrier_chunk = []
        passenger_chunk = []

        for j in range(step_size):
            index = step_size * i + j
            if index == total_volumes:
                break

            carrier_vol = carrier.dataobj[..., index]
            passenger_vol = passenger.dataobj[..., index]
            moving = ants.from_numpy(np.asarray(carrier_vol, dtype='float32'))
            psg = ants.from_numpy(np.asarray(passenger_vol, dtype='float32'))

            moco = ants.registration(fixed, moving, type_of_transform='SyN')
            carrier_chunk.append(moco['warpedmovout'].numpy())
            psg_moco = ants.apply_transforms(fixed, psg, moco['fwdtransforms'])
            passenger_chunk.append(psg_moco.numpy())

            for tmp in moco['fwdtransforms']:
                os.remove(tmp)
            for tmp in moco['invtransforms']:
                os.remove(tmp)

        carrier_chunk = np.moveaxis(np.asarray(carrier_chunk), 0, -1)
        with h5py.File(carrier_h5, 'a') as f:
            f['data'][..., steps[i]:steps[i + 1]] = carrier_chunk

        passenger_chunk = np.moveaxis(np.asarray(passenger_chunk), 0, -1)
        with h5py.File(passenger_h5, 'a') as f:
            f['data'][..., steps[i]:steps[i + 1]] = passenger_chunk

    printlog('saving .nii images')
    h5_to_nii(carrier_h5)
    h5_to_nii(passenger_h5)

    printlog(directory.split('/')[-1]+' moco finished!')


def make_empty_h5(savefile, brain_dims):
    with h5py.File(savefile, 'w') as f:
        dset = f.create_dataset('data', brain_dims, dtype='float32', chunks=True)


def h5_to_nii(h5_path):
    nii_savefile = h5_path.split('.')[0] + '.nii'
    with h5py.File(h5_path, 'r+') as h5_file:
        image_array = h5_file.get("data")[:].astype('uint16')

    nifti1_limit = (2 ** 16 / 2)
    if np.any(np.array(image_array.shape) >= nifti1_limit):  # Need to save as nifti2
        nib.save(nib.Nifti2Image(image_array, np.eye(4)), nii_savefile)
    else:  # Nifti1 is OK
        nib.save(nib.Nifti1Image(image_array, np.eye(4)), nii_savefile)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
