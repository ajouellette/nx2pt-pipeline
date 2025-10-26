import argparse
import os
from os import path
import sys
import warnings
import yaml
import numpy as np
import healpy as hp
import pymaster as nmt
import sacc
from astropy.table import Table

from .data import ClData
from .tracer import MapTracer, CatalogTracer
from .namaster_tools import compute_cls_cov
from .utils import get_ul_key, parse_cl_key, parse_tracer_bin
from .utils import Timer, preprocess_yaml


def get_tracer(nside, tracer_config):
    """Load tracer information."""
    name = tracer_config["name"]
    data_dir = tracer_config["data_dir"]
    if "healpix" in tracer_config.keys():
        tracer_type = "healpix"
    elif "catalog" in tracer_config.keys():
        tracer_type = "catalog"
    else:
        raise ValueError(f"Tracer {key} must have either a 'healpix' or 'catalog' section")
    bins = tracer_config.get("bins", 1)
    bins_one_indexed = tracer_config.get("bins_one_indexed", False)
    use_mask_squared = tracer_config.get("use_mask_squared", False)  # old option
    correct_qu_sign = tracer_config.get("correct_qu_sign", False)

    print(name, f"({bins} bins)" if bins > 1 else '')

    tracer_bins = []
    bin_inds = range(1, bins+1) if bins_one_indexed else range(bins)
    for ind, bin_i in enumerate(bin_inds):
        bin_name = name if bins == 1 else f"{name} (bin {ind})"

        if "beam" in tracer_config.keys():
            if tracer_config["beam"] == "pixwin":
                beam = hp.pixwin(nside)
            elif tracer_config["beam"] is None or tracer_config["beam"] == "None":
                beam = None
            else:
                beam_file = path.join(data_dir, tracer_config[key]["beam"].format(bin=bin_i, nside=nside))
                beam = np.loadtxt(beam_file)
        else:
            beam = np.ones(3*nside)

        if tracer_type == "healpix":
            map_file = path.join(data_dir, tracer_config["healpix"]["map"].format(bin=bin_i, nside=nside))
            maps = np.atleast_2d(hp.read_map(map_file, field=None))

            if correct_qu_sign and len(maps) == 2:
                maps = np.array([-maps[0], maps[1]])
            
            if isinstance(tracer_config["healpix"].get("mask", None), str):
                # compatibility with old mask specification
                is_masked = tracer_config["healpix"].get("is_masked", False)
                mask_file = path.join(data_dir, tracer_config["healpix"]["mask"].format(bin=bin_i, nside=nside))
                full_mask = hp.read_map(mask_file)
                if use_mask_squared: full_mask = mask**2
                if not is_masked:
                    maps *= full_mask
            else:
                # new, more flexible mask specification
                full_mask = np.ones(maps.shape[1])
                for key in tracer_config["healpix"].keys():
                    if not key.startswith("mask"):
                        continue
                    mask_options = tracer_config["healpix"][key]
                    mask_file = path.join(data_dir, mask_options["file"].format(bin=bin_i, nside=nside))
                    power = mask_options.get("power", 1)
                    included_in_map = mask_options.get("included_in_map", False)
                    mask = hp.read_map(mask_file)**power
                    full_mask *= mask
                    if not included_in_map:
                        maps *= mask

            tracer = MapTracer(bin_name, maps, full_mask, beam=beam, masked_on_input=True)

        elif tracer_type == "catalog":
            cat_file = path.join(data_dir, tracer_config["catalog"]["file"].format(bin=bin_i))
            catalog = Table.read(cat_file)
            pos = [get_ul_key(catalog, "ra"), get_ul_key(catalog, "dec")]
            try:
                weights = get_ul_key(catalog, "weight")
            except KeyError:
                warnings.warn("Did not find a 'weight' column, assuming uniform weights")
                weights = np.ones(len(catalog))
            if "fields" in tracer_config["catalog"].keys():
                fields = [catalog[f] for f in tracer_config["catalog"]["fields"]]
                if correct_qu_sign and len(fields) == 2:
                    fields = [-fields[0], fields[1]]
                pos_rand = None
                weights_rand = None
            elif "randoms" in tracer_config["catalog"].keys():
                fields = None
                rand_file = path.join(data_dir, tracer_config["catalog"]["randoms"].format(bin=bin_i))
                rand_cat = Table.read(rand_file)
                pos_rand = [get_ul_key(rand_cat, "ra"), get_ul_key(rand_cat, "dec")]
                try:
                    weights_rand = get_ul_key(rand_cat, "weight")
                except KeyError:
                    weights_rand = np.ones(len(rand_cat))
            else:
                raise ValueError(f"Must specify either fields or randoms in {tracer_key}")

            tracer = CatalogTracer(bin_name, pos, weights, 3*nside-1, fields=fields, beam=beam,
                                   pos_rand=pos_rand, weights_rand=weights_rand)
        else:
            raise ValueError("Tracer must be either a healpix field or a catalog")

        # shot noise
        if "noise_est" in tracer_config.keys():
            noise_est = tracer_config["noise_est"]
            if not isinstance(noise_est, list):
                noise_est = bins * [noise_est,]
            tracer.noise_est = noise_est[ind]

        tracer_bins.append(tracer)
    return tracer_bins


def main():
    parser = argparse.ArgumentParser(description="Run a Nx2-point analysis pipeline")
    parser.add_argument("config_file", help="YAML file specifying pipeline to run")
    parser.add_argument("--nside", type=int, default=None,
                        help="overrides nside in config file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use the workspace cache")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--low-mem", action="store_true")
    args = parser.parse_args()

    config = preprocess_yaml(args.config_file)
    config = yaml.safe_load(config)

    nside = args.nside if args.nside is not None else config["nside"]
    print("Nside", nside)

    wksp_dir = None if args.no_cache else config.get("workspace_dir", None)
    print("Using workspace cache:", wksp_dir)

    tracer_keys = list(config["tracers"].keys())
    print(f"Found {len(tracer_keys)} tracer(s)")
    tracers = dict()
    for tracer_key in tracer_keys:
        tracer_bins = get_tracer(nside, config["tracers"][tracer_key])
        print(tracer_bins)
        tracers[tracer_key] = tracer_bins

    xspec_keys = [key for key in config.keys() if key.startswith("cross_spectra")]
    print(f"Found {len(xspec_keys)} set(s) of cross-spectra to calculate")
    for xspec_key in xspec_keys:
        if "save_npz" not in config[xspec_key].keys() and \
           "save_sacc" not in config[xspec_key].keys():
            print(f"Warning! No output will be saved for the block {xspec_key}")

    for xspec_key in xspec_keys:
        if "save_sacc" in config[xspec_key].keys():
            if path.isfile(config[xspec_key]["save_sacc"]["file"].format(nside=nside)) and not args.overwrite:
                print(f"output file for {xspec_key} already exists, skipping")
                continue
        if "save_npz" in config[xspec_key].keys():
            if path.isfile(config[xspec_key]["save_npz"].format(nside=nside)) and not args.overwrite:
                print(f"output file for {xspec_key} already exists, skipping")
                continue

        # cross-spectra with their individual settings
        xspectra = config[xspec_key]["list"]

        print("Computing set", xspec_key)
        print("Tracers:", [x["tracers"] for x in xspectra])

        # apply default binning
        for xspec in xspectra:
            if "binning" not in xspec.keys():
                if "binning" in config.keys():
                    xspec["binning"] = config["binning"]
                else:
                    raise ValueError("Must specify a binning scheme: either a global default, or individually for each cross-spectrum.")

        # covariance of cross-spectra
        calc_cov = config[xspec_key].get("covariance", False)
        calc_interbin_cov = config[xspec_key].get("interbin_cov", False)

        # calculate everything
        result = compute_cls_cov(tracers, xspectra, compute_cov=calc_cov,
                                 compute_interbin_cov=calc_interbin_cov, wksp_cache=wksp_dir,
                                 keep_wksps_in_mem=not args.low_mem)

        data = ClData(**result, tracers=tracers)

        # TODO: should probably get rid of saving to npz files
        ## save all cross-spectra
        #if "save_npz" in config[xspec_key].keys():
        #    save_npz_file = config[xspec_key]["save_npz"].format(nside=nside)
        #    print("Saving to", save_npz_file)
        #    data.write_to_npz(save_npz_file)

        # create sacc file
        if "save_sacc" in config[xspec_key].keys():
            save_sacc_file = config[xspec_key]["save_sacc"]["file"].format(nside=nside)
            print("Saving to", save_sacc_file)
            metadata = config[xspec_key]["save_sacc"].get("metadata", None)
            data.write_to_sacc(save_sacc_file, metadata=metadata)
