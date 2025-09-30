import os
from os import path
import numpy as np
import healpy as hp
import pymaster as nmt
import joblib

from .utils import parse_tracer_bin, parse_cl_key
from .utils import Timer


def get_ell_bins(nside, bin_config):
    """Generate ell bins from config."""
    bpw_edges = bin_config.get("bpw_edges", None)
    if bpw_edges is None:
        kind = bin_config.get("kind", "linear")
        lmin = bin_config.get("ell_min", 2)
        lmax = bin_config.get("ell_max", 3*nside-1)
        nbpws = bin_config.get("nbpws", None)
        if nbpws is None and kind == "linear":
            delta_ell = bin_config["delta_ell"]
            nbpws = (lmax - lmin) // delta_ell
        elif nbpws is None:
            raise ValueError("Must specify nbpws for non-linear binning")
        bpw_edges = get_bpw_edges(lmin, lmax, nbpws, kind)
    nmt_bin = get_nmtbins(nside, bpw_edges)
    return nmt_bin


def get_bpw_edges(lmin, lmax, nbpws, kind):
    """Generate bandpower edges."""
    if kind == "linear":
        return np.linspace(lmin, lmax, nbpws+1, dtype=int)
    elif kind == "log":
        return np.geomspace(lmin, lmax, nbpws+1, dtype=int)
    elif kind == "sqrt":
        return (np.linspace(np.sqrt(lmin), np.sqrt(lmax), nbpws+1)**2).astype(int)
    else:
        raise ValueError("kind should be one of: linear, log, sqrt")


def get_nmtbins(nside, bpw_edges, weights=None, f_ell=None):
    """Generate NmtBin from nside and bandpower edges."""
    ells = np.arange(3*nside)
    nbpws = len(bpw_edges) - 1
    bpws = np.digitize(ells, bpw_edges) - 1
    bpws[bpws == nbpws] = -1
    return nmt.NmtBin(ells=ells, bpws=bpws, weights=weights, f_ell=f_ell)


def get_workspace(nmt_field1, nmt_field2, nmt_bins, wksp_cache=None):
    """Get the NmtWorkspace for given fields and bins (with caching)."""

    if wksp_cache is None:
        wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, nmt_bins)
        return wksp

    # hash on mask alms (to support catalog fields) and spins
    # TODO: this operation is not symmetric wrt field1/field2
    with Timer("hashing.."):
        hash_key = joblib.hash([nmt_field1.get_mask_alms(), nmt_field1.spin,
                                nmt_field2.get_mask_alms(), nmt_field2.spin])
    wksp_file = path.join(wksp_cache, f"cl/{hash_key}.fits")

    if path.isfile(wksp_file):
        # load from existing file
        with Timer("loading from file..."):
            wksp = nmt.NmtWorkspace.from_file(wksp_file)
            wksp.check_unbinned()
        # update bins and beams after loading
        with Timer("updating beams and bins..."):
            wksp.update_beams(nmt_field1.beam, nmt_field2.beam)
            wksp.update_bins(nmt_bins)
    else:
        # compute and save to file
        with Timer("computing..."):
            wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, nmt_bins)
        os.makedirs(path.dirname(wksp_file), exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def get_cov_workspace(nmt_field1a, nmt_field2a, nmt_field1b=None, nmt_field2b=None,
                      wksp_cache=None):
    """
    Get the NmtCovarianceWorkspace object needed to calculate the covariance between the
    cross-spectra (field1a, field2a) and (field1b, field2b).
    """
    if nmt_field1b is None and nmt_field2b is None:
        nmt_field1b = nmt_field1a
        nmt_field2b = nmt_field2a
    elif nmt_field1b is None or nmt_field2b is None:
        raise ValueError("Must provide either 2 or 4 fields")

    if wksp_cache is None:
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a,
                                                      nmt_field1b, nmt_field2b)
        return wksp

    # hash masks and spins
    hash_key = joblib.hash([nmt_field1a.get_mask(), nmt_field1a.spin,
                            nmt_field2a.get_mask(), nmt_field2a.spin,
                            nmt_field1b.get_mask(), nmt_field1b.spin,
                            nmt_field2b.get_mask(), nmt_field2b.spin])
    wksp_file = path.join(wksp_cache, f"cov/{hash_key}.fits")

    if path.isfile(wksp_file):
        wksp = nmt.NmtCovarianceWorkspace.from_file(wksp_file)
        print("Using cached workspace")
    else:
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a,
                                                      nmt_field1b, nmt_field2b)
        os.makedirs(path.dirname(wksp_file), exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def compute_cl(wksp_dir, nmt_field1, nmt_field2, nmt_bins, return_bpw=False):
    """Calculate the x-spectrum between tracer1 and tracer2."""
    wksp = get_workspace(wksp_dir, nmt_field1, nmt_field2, nmt_bins)
    print(wksp.wsp.lmax, wksp.wsp.lmax_mask)
    pcl = nmt.compute_coupled_cell(nmt_field1, nmt_field2)
    cl = wksp.decouple_cell(pcl)
    if return_bpw:
        return cl, wksp.get_bandpower_windows()
    return cl


def fsky(nmt_field1, nmt_field2):
    return np.mean(nmt_field1.get_mask() * nmt_field2.get_mask())


def coupled_cl_over_fsky(nmt_field1, nmt_field2):
    return nmt.compute_coupled_cell(nmt_field1, nmt_field2) / fsky(nmt_field1, nmt_field2)


def compute_gaussian_cov(wksp_dir, nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b,
                         nmt_bins):
    """Compute the Gaussian covariance between powerspectra A and B."""
    # get workspaces
    cov_wksp = get_cov_workspace(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b,
                                 wksp_cache=wksp_dir)
    wksp_a = get_workspace(nmt_field1a, nmt_field2a, nmt_bins, wksp_cache=wksp_dir)
    wksp_b = get_workspace(nmt_field1b, nmt_field2b, nmt_bins, wksp_cache=wksp_dir)

    spins = [nmt_field1a.spin, nmt_field2a.spin, nmt_field1b.spin, nmt_field2b.spin]

    # iNKA approximation: get coupled cls divded by mean of product of masks
    pcl1a1b = coupled_cl_over_fsky(nmt_field1a, nmt_field1b)
    pcl2a1b = coupled_cl_over_fsky(nmt_field2a, nmt_field1b)
    pcl1a2b = coupled_cl_over_fsky(nmt_field1a, nmt_field2b)
    pcl2a2b = coupled_cl_over_fsky(nmt_field2a, nmt_field2b)

    if np.isnan(pcl1a1b).any(): print("pcl1a1b has nans")
    if np.isnan(pcl1a2b).any(): print("pcl1a2b has nans")
    if np.isnan(pcl2a1b).any(): print("pcl2a1b has nans")
    if np.isnan(pcl2a2b).any(): print("pcl2a2b has nans")

    cov = nmt.gaussian_covariance(cov_wksp, *spins, pcl1a1b, pcl1a2b, pcl2a1b, pcl2a2b,
                                  wksp_a, wksp_b)
    if np.isnan(cov).any(): print("cov has nans")
    return cov


def compute_cls_cov(tracers, xspectra, compute_cov=True, compute_interbin_cov=True,
                    wksp_cache=None):
    """
    Calculate all cross-spectra and covariances from a list of tracers.

    Parameters:
    tracers: dict(key: Tracer) - dictionary of tracer ID, tracer pairs
    xspectra: list(dict(tracers: tracer_pair, **settings)) - list of cross-spectra to compute

    Returns: ells, cls, bpws, [covs]
    """
    result = dict(ells={}, cls={}, bpws={}, covs={}, nls={})
    wksps = dict()  # not returned, but needed internally

    # loop over all cross-spectra
    for xspec in xspectra:
        # get tracers and cross-spectrum specific settings
        tracer1_key, tracer2_key = xspec["tracers"]
        tracer1 = tracers[tracer1_key]
        tracer2 = tracers[tracer2_key]
        subtract_noise = xspec.get("subtract_noise", False)
        autos_only = xspec.get("autos_only", False)
        save_nl = xspec.get("save_nl", False)
        # get binning
        if hasattr(tracer1[0], "nside"):
            nside = tracer1[0].nside
        else:
            nside = (tracer1[0].lmax + 1) / 3
        bins = get_ell_bins(nside, xspec["binning"])

        # loop over all bins
        for i in range(len(tracer1)):
            for j in range(len(tracer2)):
                # skip duplicates
                if tracer1 == tracer2 and j < i:
                    continue
                # skip crosses if only doing autos
                if tracer1 == tracer2 and autos_only and i != j:
                    continue
                cl_key = f"{tracer1_key}_{i}, {tracer2_key}_{j}"
                with Timer(f"computing cross-spectrum {cl_key}..."):
                    with Timer("getting workspace..."):
                        wksp = get_workspace(tracer1[i].field, tracer2[j].field, bins,
                                            wksp_cache=wksp_cache)
                    # save pcls and wksps for covariance calculation
                    with Timer("computing pcl..."):
                        pcl = nmt.compute_coupled_cell(tracer1[i].field, tracer2[j].field)
                    wksps[cl_key] = wksp
                    # only subtract noise from auto-spectra
                    if subtract_noise and i == j and tracer1 == tracer2:
                        if not hasattr(tracer1[i], "noise_est"):
                            print("tracer has no shot noise estimate")
                        else:
                            print(f"subtracting noise estimate: {tracer1[i].noise_est:.4e}")
                            # only subtract from EE, BB if spin-2
                            if tracer1[i].spin == 0:
                                pcl -= tracer1[i].noise_est
                            else:
                                pcl[0][tracer1[i].spin:] -= tracer1[i].noise_est
                                pcl[-1][tracer1[i].spin:] -= tracer1[i].noise_est
                    cl = wksp.decouple_cell(pcl)
                    if save_nl:
                        # save nl templates (decoupled unit amplitude)
                        nl = np.zeros_like(pcl)
                        nl[0][tracer1[i].spin:] = 1
                        nl[-1][tracer1[i].spin:] = 1
                        nl = wksp.decouple_cell(nl)
                        result["nls"][cl_key] = nl
                    # save quantities
                    result["ells"][cl_key] = bins.get_effective_ells()
                    result["cls"][cl_key] = cl
                    result["bpws"][cl_key] = wksp.get_bandpower_windows()

    if not compute_cov:
        return result

    # loop over all covariances
    cl_keys = list(result["cls"].keys())
    for i in range(len(cl_keys)):
        cl_key_a = cl_keys[i]
        (tracer_a1_key, bin_a1), (tracer_a2_key, bin_a2) = parse_cl_key(cl_key_a)
        field_a1 = tracers[tracer_a1_key][bin_a1].field
        field_a2 = tracers[tracer_a2_key][bin_a2].field

        # skip covariances that involve a catalog field
        if tracers[tracer_a1_key][bin_a1].is_cat_field or \
           tracers[tracer_a2_key][bin_a2].is_cat_field:
            print("Skipping covariances involving catalog field (not implemented yet)")
            continue

        for j in range(i, len(cl_keys)):
            cl_key_b = cl_keys[j]

            if not compute_interbin_cov:
                # skip all off-diagonal covs
                if cl_key_a != cl_key_b:
                    continue

            (tracer_b1_key, bin_b1), (tracer_b2_key, bin_b2) = parse_cl_key(cl_key_b)
            field_b1 = tracers[tracer_b1_key][bin_b1].field
            field_b2 = tracers[tracer_b2_key][bin_b2].field

            # skip covariances that involve a catalog field
            if tracers[tracer_b1_key][bin_b1].is_cat_field or \
               tracers[tracer_b2_key][bin_b2].is_cat_field:
                print("Skipping covariances involving catalog field (not implemented yet)")
                continue

            cov_key = f"{cl_key_a}, {cl_key_b}"
            print("computing covariance", cov_key)
            cov_wksp = get_cov_workspace(field_a1, field_a2, field_b1, field_b2,
                                         wksp_cache=wksp_cache)
            wksp_a = wksps[cl_key_a]
            wksp_b = wksps[cl_key_b]
            spins = [field_a1.spin, field_a2.spin, field_b1.spin, field_b2.spin]
            pcl_a1b1 = coupled_cl_over_fsky(field_a1, field_b1)
            pcl_a1b2 = coupled_cl_over_fsky(field_a1, field_b2)
            pcl_a2b1 = coupled_cl_over_fsky(field_a2, field_b1)
            pcl_a2b2 = coupled_cl_over_fsky(field_a2, field_b2)

            cov = nmt.gaussian_covariance(cov_wksp, *spins, pcl_a1b1, pcl_a1b2,
                                          pcl_a2b1, pcl_a2b2, wksp_a, wksp_b)
            if np.isnan(cov).any(): print("cov has nans")
            result["covs"][cov_key] = cov

    return result
