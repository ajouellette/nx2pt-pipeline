import datetime
import os
from os import path
import warnings
import numpy as np
from scipy import interpolate
import sacc


class ClInterpolator(interpolate.CubicSpline):
    """Interpolate power spectra in log-log space."""

    def __init__(self, ell, cl, axis=0, log_ell=True, log_cl=False):
        self.log_ell = log_ell
        self.log_cl = log_cl
        x = ell if not log_ell else np.log(ell)
        y = cl if not log_cl else np.log(cl)
        super().__init__(x, y, axis=axis)

    def __call__(self, ell):
        x = ell if not self.log_ell else np.log(ell)
        y = super().__call__(x)
        if self.log_cl:
            return np.exp(y)
        return y


def bin_theory_cl(theory_cl, bpws, ell=None, fix_dipole=True, fix_monopole=True, fill=False):
    """Bin a theory Cl given some bandpower windows."""
    nells = bpws.shape[1]
    if ell is not None:
        interp = ClInterpolator(ell, theory_cl)
        if fix_dipole:
            theory_cl = np.hstack([[0, 0], interp(np.arange(2, nells))])
        elif fix_monopole:
            theory_cl = np.hstack([[0,], interp(np.arange(1, nells))])
        else:
            theory_cl = interp(np.arange(nells))
    if len(theory_cl) < nells:
        if fill:
            theory_cl = np.hstack([theory_cl, np.zeros(nells - len(theory_cl))])
        else:
            raise ValueError("theory Cl has fewer ells than the bandpower windows.")
    return np.dot(bpws, theory_cl[:nells])


def get_cl_dtypes(ncls):
    """Return the sacc data types that correspond to a given number of Cls."""
    if ncls == 1:
        return ["cl_00"]
    if ncls == 2:
        return ["cl_0e", "cl_0b"]
    if ncls == 4:
        return ["cl_ee", "cl_eb", "cl_be", "cl_bb"]
    raise ValueError("ncls must be 1, 2, or 4")


class ClData:

    def __init__(self, ells, cls, bpws={}, covs={}, nls={}, tracers=None):
        self.ells = ells
        self.cls = cls
        self.bpws = bpws
        self.covs = covs
        self.nls = nls
        if tracers is not None:
            self.tracer_info = dict()
            for key, tracer_bins in tracers.items():
                for i, tracer_bin in enumerate(tracer_bins):
                    tracer_info = dict(description=tracer_bin.name, spin=tracer_bin.spin)
                    self.tracer_info[f"{key}_{i}"] = tracer_info
        else:
            self.tracer_info = None

    @property
    def tracers(self):
        tracers = set()
        for key in self.cls.keys():
            cl_tracers = key.split(", ")
            tracers.add(cl_tracers[0])
            tracers.add(cl_tracers[1])
        return sorted(list(tracers))

    @property
    def tracer_pairs(self):
        tracer_pairs = []
        for key in self.cls.keys():
            cl_tracers = key.split(", ")
            tracer_pairs.append(tuple(cl_tracers))
        return tracer_pairs

    @classmethod
    def from_npz(cls, filename):
        """Load data from a .npz file."""
        cls = dict()
        covs = dict()
        bpws = dict()
        with np.load(filename) as f:
            for key in f.keys():
                if key.startswith("cl_"):
                    cls[key.removeprefix("cl_")] = f[key]
                elif key.startswith("cov_"):
                    covs[key.removeprefix("cov_")] = f[key]
                elif key.startswith("bpw_"):
                    bpws[key.removeprefix("bpw_")] = f[key]
            ell_eff = f["ell_eff"]
        return Data(ell_eff, cls, covs, bpws)

    @classmethod
    def from_sacc(cls, filename):
        pass

    @classmethod
    def from_theory_cls(cls, theory_cls, covs, bpws):
        pass

    def get_cl(self, tracer1, tracer2=None, dtype=None):
        if tracer2 is None:
            tracer2 = tracer1
        cl_key1 = f"{tracer1}, {tracer2}"
        cl_key2 = f"{tracer2}, {tracer1}"
        try:
            cl = self.cls[cl_key1]
        except KeyError:
            try:
                cl = self.cls[cl_key2]
            except KeyError:
                raise KeyError(f"could not find Cl for tracers {tracer1} and {tracer2}")
        if dtype is None:
            return cl
        ind = get_cl_dtypes(len(cl)).index(dtype)
        return cl[ind]


    def get_cov(self, cl1, cl2=None, dtype1=None, dtype2=None):
        """Get covariance of cl1 and cl2."""
        if cl2 is None:
            cl2 = cl1
        cov_key1 = f"{cl1}, {cl2}"
        cov_key2 = f"{cl2}, {cl1}"
        try:
            cov = self.covs[cov_key1]
        except KeyError:
            try:
                # need to transpose if switching order
                cov = self.covs[cov_key2].T
            except KeyError:
                raise KeyError(f"could not find Cov for Cls {cl1} and {cl2}")
        if dtype1 is None and dtype2 is None:
            return cov
        if dtype2 is None:
            dtype2 = dtype1
        ncls = [len(self.cls[cl1]), len(self.cls[cl2])]
        nbpws = [len(self.ells[cl1]), len(self.ells[cl2])]
        ind1 = get_cl_dtypes(ncls[0]).index(dtype1)
        ind2 = get_cl_dtypes(ncls[1]).index(dtype2)
        cov_shape = (nbpws[0], ncls[0], nbpws[1], ncls[1])
        return cov.reshape(cov_shape)[:,ind1,:,ind2]

    def build_full_cov_e(self, cls, scale_cuts=None, fill_off_diag=True):
        covs = []
        if scale_cuts is not None:
            use_ell = (self.ell_eff >= scale_cuts[0]) * (self.ell_eff < scale_cuts[1])
        else:
            use_ell = np.ones(self.nbpws, dtype=bool)
        for i in range(len(cls)):
            covs_i = []
            for j in range(len(cls)):
                if i == j:
                    cov = self.get_cov(cls[i], cls[i])[:,0,:,0]
                else:
                    try:
                        cov = self.get_cov(cls[i], cls[j])[:,0,:,0]
                    except KeyError:
                        cov = np.zeros((self.nbpws, self.nbpws))
                covs_i.append(cov[use_ell][:,use_ell])
            covs.append(covs_i)
        return np.block(covs)

    def write_to_npz(self, filename):
        """Save cross-spectra, covariances, and bandpower windows to a .npz file."""
        save_dict = {"cl_" + cl_key: self.cls[cl_key] for cl_key in self.cls.keys()} | \
                    {"cov_" + cov_key: self.covs[cov_key] for cov_key in self.covs.keys()} | \
                    {"bpw_" + cl_key: self.bpws[cl_key] for cl_key in self.cls.keys()} | \
                    {"ell_eff": self.ell_eff}
        np.savez(filename, **save_dict)

    def write_to_sacc(self, filename, metadata=None, overwrite=True):
        """Create a sacc fits file containing the cross-spectra and covariance."""
        s = sacc.Sacc()
        # metadata
        s.metadata["creation"] = datetime.date.today().isoformat()
        if metadata is not None:
            for key in metadata.keys():
                s.metadata[key] = metadata[key]
        # tracers (currently only save as misc tracers)
        for tracer in self.tracers:
            if self.tracer_info is not None:
                tracer_meta = self.tracer_info.get(tracer, None)
            else:
                tracer_meta = None
            s.add_tracer("Misc", tracer, metadata=tracer_meta)
        # data
        for tracer1, tracer2 in self.tracer_pairs:
            cl_key = f"{tracer1}, {tracer2}"
            ells = self.ells[cl_key]
            cl = self.cls[cl_key]
            if self.bpws == {}:
                warnings.warn("Data has no bandpower information")
                bpws = [None for i in range(len(cl))]
            else:
                nmt_bpws = self.bpws[cl_key]
                ell = np.arange(nmt_bpws.shape[-1])
                bpws = [sacc.BandpowerWindow(ell, nmt_bpws[i,:,i,:].T) for i in range(len(cl))]
            # possible spin combinations
            for i, dtype in enumerate(get_cl_dtypes(len(cl))):
                s.add_ell_cl(dtype, tracer1, tracer2, ells, cl[i], window=bpws[i])
        # save noise templates as tags on the corresponding cls
        if self.nls != {}:
            for cl_key, nl in self.nls.items():
                tracers = cl_key.split(", ")
                for i, dtype in enumerate(get_cl_dtypes(len(nl))):
                    inds = s.indices(data_type=dtype, tracers=tracers)
                    for nl_i, ind in enumerate(inds):
                        s.data[ind].tags["nl"] = nl[i][nl_i]
        # covariance
        if self.covs == {}:
            warnings.warn("Data has no covariance information")
        else:
            if len(self.cls.keys()) == len(self.covs.keys()):
                # block diagonal covariance
                s.add_covariance([self.get_cov(cl_key, cl_key, reshape_cov=False) for cl_keys in self.cls.keys()])
            else:
                # loop over all possible cross-spectra
                full_cov = np.zeros((len(s.mean), len(s.mean)))
                for tracers1 in s.get_tracer_combinations():
                    for dtype1 in s.get_data_types(tracers1):
                        inds1 = s.indices(tracers=tracers1, data_type=dtype1)
                        for tracers2 in s.get_tracer_combinations():
                            for dtype2 in s.get_data_types(tracers2):
                                inds2 = s.indices(tracers=tracers2, data_type=dtype2)
                                cov = self.get_cov(", ".join(tracers1), ", ".join(tracers2), dtype1=dtype1, dtype2=dtype2)
                                full_cov[np.ix_(inds1, inds2)] = cov
                s.add_covariance(full_cov)
        # write
        if path.dirname(filename):
            os.makedirs(path.dirname(filename), exist_ok=True)
        s.save_fits(filename, overwrite=overwrite)
