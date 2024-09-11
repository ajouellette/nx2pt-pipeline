import os
import sys
import yaml
import numpy as np
import healpy as hp
import pymaster as nmt
import joblib
import sacc

import nx2pt


def get_ell_bins(config):
    """Generate ell bins from config."""
    nside = config["nside"]
    ell_min = config["ell_min"]
    dl = config["delta_ell"]
    ell_bins = np.linspace(ell_min, 3*nside, int((3*nside - ell_min) / dl) + 1, dtype=int)
    return ell_bins


def get_tracer(config, key):
    """Load tracer information."""
    nside = config["nside"]
    name = config[key]["name"]
    data_dir = config[key]["data_dir"]
    if "bins" in config[key].keys():
        bins = config[key]["bins"]
    else:
        bins = 1
    if "use_mask_squared" in config[key].keys():
        use_mask_squared = config[key]["use_mask_squared"]
    else:
        use_mask_squared = False
    if "correct_qu_sign" in config[key].keys():
        correct_qu_sign = config[key]["correct_qu_sign"]
    else:
        correct_qu_sign = False

    print(name, f"({bins} bins)" if bins > 1 else '')

    tracer_bins = []
    for bin in range(bins):
        bin_name = name if bins == 1 else f"{name} (bin {bin})"
        map_file = data_dir + '/' + config[key]["map"].format(bin=bin, nside=nside)
        mask_file = data_dir + '/' + config[key]["mask"].format(bin=bin, nside=nside)
        if "beam" in config[key].keys():
            beam_file = data_dir + '/' + config[key]["beam"].format(bin=bin, nside=nside)
        else:
            beam = np.ones(3*nside)

        maps = np.atleast_2d(hp.read_map(map_file, field=None))
        if correct_qu_sign and len(maps) == 2:
            maps = np.array([-maps[0], maps[1]])

        mask = hp.read_map(mask_file)
        if use_mask_squared: mask = mask**2

        nmt_field = nmt.NmtField(mask, maps, beam=beam)
        tracer = dict(name=bin_name, nmt_field=nmt_field)
        tracer_bins.append(tracer)
        print(bin_name, f"spin {nmt_field.spin}")

    return tracer_bins


def parse_tracer_bin(tracer_bin_key):
    """Takes a string of the form tracer_name_{int} and returns tracer_name, int."""
    key_split = tracer_bin_key.split('_')
    tracer_name = '_'.join(key_split[:-1])
    tracer_bin = int(key_split[-1])
    return tracer_name, tracer_bin


def save_sacc(config):
    pass


def save_npz(file_name, ell_eff, cls, covs, bpws):
    """Save cross-spectra, covariances, and bandpower windows to a .npz file."""
    assert bpws.keys() == cls.keys(), "Each cross-spectrum should have a corresponding bandpower window"
    save_dict = {"cl_" + str(cl_key): cls[cl_key] for cl_key in cls.keys()} | \
                {"cov_" + str(cov_key): covs[cov_key] for cov_key in covs.keys()} | \
                {"bpw_" + str(cl_key): bpws[cl_key] for cl_key in cls.keys()} | \
                {"ell_eff": ell_eff}
    np.savez(save_npz_file, **save_dict)


def main():
    with open(sys.argv[1]) as f:
        config = yaml.full_load(f)

    print(config)

    tracer_keys = [key for key in config.keys() if key.startswith("tracer")]
    print(f"Found {len(tracer_keys)} tracers")
    tracers = dict()
    for tracer_key in tracer_keys:
        tracer = get_tracer(config, tracer_key)
        tracers[tracer_key] = tracer

    xspec_keys = [key for key in config.keys() if key.startswith("cross_spectra")]
    print(f"Found {len(xspec_keys)} set(s) of cross-spectra to calculate")
    for xspec_key in xspec_keys:
        if "save_npz" not in config[xspec_key].keys() and "save_sacc" not in config[xspec_key].keys():
            print(f"Warning! No output will be saved for the block {xspec_key}")

    ell_bins = get_ell_bins(config)
    nmt_bins = nmt.NmtBin.from_edges(ell_bins[:-1], ell_bins[1:])
    ell_eff = nmt_bins.get_effective_ells()
    print(f"Will calculate {len(ell_eff)} bandpowers between ell = {ell_bins[0]} and ell = {ell_bins[-1]}")
    wksp_dir = config["workspace_dir"]

    for xspec_key in xspec_keys:
        xspec_list = config[xspec_key]["list"]
        print("Computing set", xspec_list)

        calc_cov = False
        calc_interbin_cov = False
        if "covariance" in config[xspec_key].keys():
            calc_cov = config[xspec_key]["covariance"]
        if "interbin_cov" in config[xspec_key].keys():
            calc_interbin_cov = config[xspec_key]["interbin_cov"]

        cls = dict()
        bpws = dict()
        # compute each cross-spectrum in the set
        for xspec in xspec_list:
            if xspec[0] not in tracer_keys or xspec[1] not in tracer_keys:
                raise ValueError(f"Undefined tracer in x-spectrum {xspec}")

            tracer1 = tracers[xspec[0]]
            tracer2 = tracers[xspec[1]]

            # loop over all tracer bins:
            for i in range(len(tracer1)):
                for j in range(len(tracer2)):
                    cl_key = (xspec[0]+f"_{i}", xspec[1]+f"_{j}")
                    print("Computing cross-spectrum", cl_key)
                    cl, bpw = nx2pt.compute_cl(wksp_dir, tracer1[i]["nmt_field"], tracer2[j]["nmt_field"], nmt_bins, return_bpw=True)
                    cls[cl_key] = cl
                    bpws[cl_key] = bpw

        # compute covariance for all pairs of cross-spectra in set
        covs = dict()
        cl_keys = list(cls.keys())
        # double loop over cl_keys
        for i in range(len(cl_keys)):
            cl_key1 = cl_keys[i]
            tracer1, bin1 = parse_tracer_bin(cl_key1[0])
            tracer2, bin2 = parse_tracer_bin(cl_key1[1])
            nmt_field1a = tracers[tracer1][bin1]["nmt_field"]
            nmt_field2a = tracers[tracer2][bin2]["nmt_field"]
            for j in range(i, len(cl_keys)):
                cl_key2 = cl_keys[j]
                cov_key = (cl_key1[0], cl_key1[1], cl_key2[0], cl_key2[1])
                print("Computing covariance", cov_key)
                tracer1, bin1 = parse_tracer_bin(cl_key2[0])
                tracer2, bin2 = parse_tracer_bin(cl_key2[1])
                nmt_field1b = tracers[tracer1][bin1]["nmt_field"]
                nmt_field2b = tracers[tracer2][bin2]["nmt_field"]

                cov = nx2pt.compute_gaussian_cov(wksp_dir, nmt_field1a, nmt_field2a,
                                                nmt_field1b, nmt_field2b, nmt_bins)
                covs[cov_key] = cov

        # save all cross-spectra
        if "save_npz" in config[xspec_key].keys():
            save_npz_file = config[xspec_key]["save_npz"].format(nside=config["nside"])
            print("Saving to", save_npz_file)
            save_npz(save_npz_file, ell_eff, cls, covs, bpws)

        # create sacc file
        #if "save_sacc" in config.keys():
            #print("Creating sacc file")
