import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from ost import io
from ost.geom import MinDistance, Vec3List
from scipy.stats import ecdf, spearmanr

logger = logging.getLogger(__name__)

class BaseAnalyzer(object):
    def __init__(self, EC_threshold=0.7, group_name="couplings", distance_cutoff=8, min_seq_length=1):
        self.distance_cutoff = distance_cutoff
        self.EC_threshold = EC_threshold
        self.min_seq_length = min_seq_length
        self.group_name = group_name

    @staticmethod
    def _to_one_based_labels(labels):
        idx = pd.Index(labels)
        if idx.empty:
            return idx
        numeric = pd.to_numeric(idx, errors="coerce")
        if numeric.isna().any():
            return idx
        vals = numeric.to_numpy(dtype=int)
        if vals.min() == 0:
            vals = vals + 1
        return pd.Index(vals)

    @classmethod
    def _ensure_one_based_df(cls, df):
        out = df.copy()
        out.index = cls._to_one_based_labels(out.index)
        out.columns = cls._to_one_based_labels(out.columns)
        return out

    @staticmethod
    def _mirror_upper_to_lower(df):
        arr = df.to_numpy(copy=True)
        lower = np.tril_indices_from(arr, k=-1)
        arr[lower] = arr.T[lower]
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    def AnalyzeProteinInterface(self, chainAhandle, chainBhandle, cutoff=8):
        if not (chainAhandle.valid and chainBhandle.valid): return pd.DataFrame()
        positions = {}
        for residue in chainAhandle.residues:
            positions[residue] = Vec3List([atom.pos for atom in residue.atoms])
        for residue in chainBhandle.residues:
            positions[residue] = Vec3List([atom.pos for atom in residue.atoms])
        surfaceAlist = []
        surfaceBlist = []
        for Aresidue in chainAhandle.residues:
            for Bresidue in chainBhandle.residues:
                minDistance = MinDistance(positions[Aresidue], positions[Bresidue])
                if minDistance < cutoff: 
                    if Aresidue not in surfaceAlist: surfaceAlist.append(Aresidue)
                    if Bresidue not in surfaceBlist: surfaceBlist.append(Bresidue)
        if not (surfaceAlist and surfaceBlist): return pd.DataFrame()
        df = pd.DataFrame(index=[f"{residue.number}" for residue in surfaceAlist], 
                        columns=[f"{residue.number}" for residue in surfaceBlist])
        for i, row in enumerate(surfaceAlist):
            for j, col in enumerate(surfaceBlist):
                df.iloc[i, j] = round(MinDistance(positions[row], positions[col]), 1)
        return df

    def ParsePLMCOutput(self, ECfile):
        long_df = pd.read_csv(ECfile, sep="\s+", header=None, names=["pos1", "aa1", "pos2", "aa2", "dummy", "score"])
        positions = np.sort(
            pd.unique(pd.concat([long_df["pos1"], long_df["pos2"]], ignore_index=True))
        )
        df = pd.DataFrame(0, index=positions, columns=positions, dtype=float)
        diff = np.diff(long_df['pos2'])
        decrease_indices = np.where(diff < 0)[0]
        df.iloc[0,1:] = long_df.iloc[0:decrease_indices[0] + 1]["score"].to_numpy()
        for i in range(len(decrease_indices)-1):
            df.iloc[i+1, i+2:] = long_df.iloc[decrease_indices[i] + 1:decrease_indices[i + 1] + 1]["score"].to_numpy()
        df.iloc[i+2, i+3:] = long_df.iloc[decrease_indices[i+1]+1:decrease_indices[i+1]+3]["score"].to_numpy()
        df.iloc[i+3, i+4] = long_df.iloc[-1]["score"]
        return self._ensure_one_based_df(self._mirror_upper_to_lower(df))

    def ParseCSVOutput(self, ECfile):
        # CSV format: res1,res2,score
        long_df = pd.read_csv(ECfile)
        if long_df.empty:
            return {}

        required = {"res1", "res2", "score"}
        if not required.issubset(long_df.columns):
            raise ValueError(f"CSV must contain columns {sorted(required)} (got {list(long_df.columns)})")

        long_df["res1"] = long_df["res1"].astype(int)
        long_df["res2"] = long_df["res2"].astype(int)
        long_df["score"] = long_df["score"].astype(float)

        positions = np.sort(
            pd.unique(pd.concat([long_df["res1"], long_df["res2"]], ignore_index=True))
        )

        df = pd.DataFrame(np.nan, index=positions, columns=positions, dtype=float)
        row_idx = np.searchsorted(positions, long_df["res1"].to_numpy())
        col_idx = np.searchsorted(positions, long_df["res2"].to_numpy())
        df_values = df.to_numpy(copy=False)
        df_values[row_idx, col_idx] = long_df["score"].to_numpy()
        return self._ensure_one_based_df(self._mirror_upper_to_lower(df))

    def ParseNumpyOutput(self, ECfile):
        arr = np.load(ECfile, allow_pickle=True)
        data = pd.DataFrame(arr)
        return self._ensure_one_based_df(data)

    def ParseZarrOutput(self, ECfile, group_name="couplings"):
        try:
            grp = zarr.open_group(ECfile, mode='r')
        except Exception as e:
            logger.error(f"Error opening Zarr file {ECfile}: {e}")
            return pd.DataFrame(), None
        saved_columns = grp["align"].attrs.get('saved_columns', None)
        arr = grp[group_name][:]
        data = pd.DataFrame(arr)
        data = self._ensure_one_based_df(data)
        return data, saved_columns

    def TruePositiveCount(self, interface, couplings):
        if not couplings:
            return 0
        TP = 0
        for pos1, pos2 in couplings.keys():
            if f'{pos1}' in interface.index and f'{pos2}' in interface.columns:
                if interface.loc[f'{pos1}', f'{pos2}'] < self.distance_cutoff:
                    TP += 1
        return TP

    def TruePositiveRatio(self, baseline_cp, new_cp):
        return new_cp / baseline_cp

    def TruePositiveRate(self, nTP, nCouplings):
        if not nCouplings:
            return 0
        return nTP / nCouplings

    def BenchmarkIndex(self, TPRfull, TPR):
        return pow(TPR,0.8)*pow(TPRfull, 0.2)

    def SpearmanCorrelation(self, baseline_cp, new_cp):
        if not baseline_cp or not new_cp:
            return 0
        if len(baseline_cp) != len(new_cp):
            print("The lengths of the two coupling score lists do not match.")
            common_pairs = set(baseline_cp) & set(new_cp)
            baseline_cp = {p: baseline_cp[p] for p in common_pairs}
            new_cp = {p: new_cp[p] for p in common_pairs}
        baseline_scores = [score for _, score in baseline_cp.items()]
        new_scores = [score for _, score in new_cp.items()]
        res = spearmanr(baseline_scores, new_scores)
        return res.statistic, res.pvalue

    def EmpiricalDistribution(self, couplings):
        res = ecdf(couplings)
        return res

class MonomerAnalyzer(BaseAnalyzer):
    def __init__(self, ECfile_baseline=None, group_name="couplings", structure_file=None, chain_name=None, EC_threshold=0.7, distance_cutoff=8, min_seq_length=1):
        super().__init__(EC_threshold, group_name, distance_cutoff, min_seq_length)
        self.nTP = None
        self.positive_couplings = None
        if structure_file is not None:
            self.ReadStructure(structure_file, chain_name)
        if ECfile_baseline is not None:
            self.ReadCouplings(ECfile_baseline)

    def ReadStructure(self, structure_file, chain_name):
        self.mol = io.LoadEntity(structure_file).Select('peptide=true')
        chain_handle = self.mol.FindChain(chain_name)
        self.interface = self.AnalyzeProteinInterface(chain_handle, chain_handle, self.distance_cutoff)

    def ReadCouplings(self, ECfile_baseline):
        if ECfile_baseline.endswith('.csv'):
            self.raw_couplings = self.ParseCSVOutput(ECfile_baseline)
        elif ECfile_baseline.endswith('.txt'):
            self.raw_couplings = self.ParsePLMCOutput(ECfile_baseline)
        elif ECfile_baseline.endswith('.npy'):
            self.raw_couplings = self.ParseNumpyOutput(ECfile_baseline)
        elif isinstance(ECfile_baseline, pd.DataFrame):
            self.raw_couplings = ECfile_baseline
        elif isinstance(ECfile_baseline, np.ndarray):
            self.raw_couplings = pd.DataFrame(ECfile_baseline)
        else:
            self.raw_couplings, _ = self.ParseZarrOutput(ECfile_baseline, self.group_name)

    def TruePositiveCount(self):
        if not self.nTP:
            self.nTP = super().TruePositiveCount(self.interface, self.positive_couplings)
        return self.nTP

    def TruePositiveRate(self):
        if not self.nTP:
            self.nTP = self.TruePositiveCount()
        nCouplings = len(self.positive_couplings)
        if nCouplings == 0: return 0
        return super().TruePositiveRate(self.nTP, nCouplings)
    
    def SpearmanCorrelation(self, query_analyzer):
        """
        Return the Spearman correlation between the baseline and query couplings.

        Parameters
        ----------
        `query_analyzer` — MonomerAnalyzer
            The query analyzer instance to compare against.

        Returns
        -------
        Tuple[float, float]
            The Spearman correlation coefficient and p-value.
        """
        def _df_to_pair_dict(df):
            dfn = df.apply(pd.to_numeric, errors="coerce")
            stacked = dfn.stack(future_stack=True)
            return {(i, j): float(v) for (i, j), v in stacked.items() if np.isfinite(v)}

        ranked_baseline = dict(
            sorted(
                _df_to_pair_dict(self.raw_couplings).items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        )
        ranked_query = dict(
            sorted(
                _df_to_pair_dict(query_analyzer.raw_couplings).items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        )
        corr, pval = super().SpearmanCorrelation(ranked_baseline, ranked_query)
        return float(corr), float(pval)

    def EmpiricalDistribution(self):
        scores = list(self.raw_couplings.values())
        res = super().EmpiricalDistribution(scores)
        return res

    def GetBenchmarkIndex(self, query_analyzer):
        if type(self.interface) == None or not self.positive_couplings: raise RuntimeError('No interface or couplings found')
        if not isinstance(query_analyzer, MonomerAnalyzer):
            raise TypeError('The input must be an instance of MonomerAnalyzer')

        couplings_query = query_analyzer.positive_couplings

        nTP_Query = super().TruePositiveCount(self.interface, couplings_query)
        TPRfull = self.TruePositiveRatio(self.nTP, nTP_Query)
        TPR = super().TruePositiveRate(nTP_Query, len(couplings_query))
        BI = self.BenchmarkIndex(TPRfull, TPR)
        Spearman = self.SpearmanCorrelation(query_analyzer)
        print(f'TP Baseline: {self.nTP};\tTP Query: {nTP_Query};\tTPRfull (%): {TPRfull:.2%};\tTPR (%): {TPR:.2%};\tBI: {BI:.3f};\tSpearman: {Spearman[0]:.3f}')
        return {'TP Baseline': self.nTP, 'TP Query': nTP_Query, 'TPRfull': TPRfull, 'TPR': TPR, 'BI': BI, 'Spearman': Spearman[0]}

    def PlotCouplings(self):
        plt.figure(figsize=(10, 6))

        scores_fp = []  # False positives (red)
        distances_fp = []
        scores_tp = []  # True positives (green)
        distances_tp = []
        scores_other = []  # Others (black)
        distances_other = []

        for pair, score in self.raw_couplings.items():
            pos1, pos2 = pair
            if abs(pos1 - pos2) >= self.min_seq_length:
                distance = self.interface.loc[f'{pos1}', f'{pos2}']
                if f'{pos1}' in self.interface.index and f'{pos2}' in self.interface.columns:
                    if score > self.EC_threshold and distance > self.distance_cutoff:
                        scores_fp.append(score)
                        distances_fp.append(distance)
                    elif score > self.EC_threshold and distance < self.distance_cutoff:
                        scores_tp.append(score)
                        distances_tp.append(distance)
                else:
                    scores_other.append(score)
                    distances_other.append(distance)

        plt.scatter(scores_other, distances_other, marker='.', alpha=0.7, color='black', label='Negatives')
        plt.scatter(scores_fp, distances_fp, marker='.', alpha=0.7, color='red', label='False Positives')
        plt.scatter(scores_tp, distances_tp, marker='.', alpha=0.7, color='green', label='True Positives')
        plt.axhline(y=self.distance_cutoff, color='r', linestyle='--', label=f'Distance cutoff ({self.distance_cutoff} Å)')
        plt.axvline(x=self.EC_threshold, color='g', linestyle='--', label=f'EC threshold ({self.EC_threshold})')

        plt.xlabel('Evolutionary Coupling Score')
        plt.ylabel('Distance (Å)')
        plt.title('Evolutionary Coupling Score vs. Residue Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set fixed x-axis range from -0.1 to 1.8
        plt.xlim(-0.1, 1.8)

        return plt.gcf()

class LegacyComplexAnalyzer(BaseAnalyzer):
    def __init__(self, ECfile_baseline=None):
        super().__init__()
        

class ComplexAnalyzer(BaseAnalyzer):
    def __init__(self, ECfile_baseline=None, group_name="couplings", saved_columns=None, EC_threshold=0.7, structure_file=None, chainA_name="A", chainA_length=None, chainB_name=None, distance_cutoff=8, switch_order=False):
        super().__init__(EC_threshold, group_name, distance_cutoff)
        self.nTP = None
        self.positive_couplings = pd.DataFrame()
        self.saved_columns = saved_columns
        self.chainA_length = chainA_length
        if structure_file is not None:
            self.ReadStructure(structure_file, chainA_name, chainB_name)
        if ECfile_baseline is not None:
            self.ReadCouplings(ECfile_baseline, switch_order=switch_order)

    def ReadStructure(self, structure_file, chainA_name, chainB_name):
        self.mol = io.LoadEntity(structure_file).Select('peptide=true')
        chainA_handle = self.mol.FindChain(chainA_name)
        chainB_handle = self.mol.FindChain(chainB_name)
        self.interface = self.AnalyzeProteinInterface(chainA_handle, chainB_handle, self.distance_cutoff)

    def ReadCouplings(self, ECfile_baseline, switch_order=False):
        # Python object as input: either a DataFrame or a numpy array
        if isinstance(ECfile_baseline, pd.DataFrame):
            self.raw_couplings = self._ensure_one_based_df(ECfile_baseline)
            self.interprotein_couplings = self._ensure_one_based_df(ECfile_baseline)
        elif isinstance(ECfile_baseline, np.ndarray):
            self.raw_couplings = self._ensure_one_based_df(pd.DataFrame(ECfile_baseline))
            self.AnalyzeInterProteinCouplings(chainA_length=self.chainA_length, saved_columns=self.saved_columns)
        # File input: determine the format based on the file extension
        else:
            if ECfile_baseline.endswith('.csv'):
                self.raw_couplings = self.ParseCSVOutput(ECfile_baseline)
            elif ECfile_baseline.endswith('.txt'):
                self.raw_couplings = self.ParsePLMCOutput(ECfile_baseline)
            elif ECfile_baseline.endswith('.npy'):
                self.raw_couplings = self.ParseNumpyOutput(ECfile_baseline)
            else:
                # Zarr format has no extension, we treat it as the default case
                self.raw_couplings, self.saved_columns = self.ParseZarrOutput(ECfile_baseline, self.group_name)
            self.raw_couplings = self._ensure_one_based_df(self.raw_couplings)
            self.AnalyzeInterProteinCouplings(chainA_length=self.chainA_length, saved_columns=self.saved_columns)
        self.interprotein_couplings = self._ensure_one_based_df(self.interprotein_couplings)
        if switch_order:
            self.interprotein_couplings = self.interprotein_couplings.T
        self.AnalyzePositiveCouplings()

    def AnalyzeInterProteinCouplings(self, chainA_length=None, saved_columns=None):
        # Determine inter-protein extraction method:
        # chainA_length takes precedence; then an explicit saved_columns; then columns
        # embedded in the zarr file; raise only if none of these are available.
        if chainA_length is None:
            raise ValueError("Chain A length must be provided")
        inter = self.GetInterProtCouplingsByChainALength(chainA_length)
        if saved_columns is not None:
            inter = self.FilterMaskedBySavedColumns(saved_columns, source_df=inter)
        self.interprotein_couplings = self._ensure_one_based_df(inter)

    def AnalyzePositiveCouplings(self):
        inter = self.interprotein_couplings.apply(pd.to_numeric, errors="coerce")
        self.positive_couplings = inter.where(inter >= self.EC_threshold)
        if self.positive_couplings.notna().sum().sum() == 0:
            self.positive_couplings = pd.DataFrame()
            print(f'No couplings found under the threshold {self.EC_threshold}')

    def GetInterProtCouplingsByChainALength(self, chainA_length):
        if chainA_length is None:
            raise ValueError('chainA_length must be provided')
        idx_vals = pd.to_numeric(self.raw_couplings.index, errors='coerce')
        col_vals = pd.to_numeric(self.raw_couplings.columns, errors='coerce')
        if idx_vals.isna().any() or col_vals.isna().any():
            raise RuntimeError('Raw couplings index/columns must be numeric for chainA_length slicing')

        # Use literal label values to split chain A and chain B.
        row_mask = idx_vals <= chainA_length
        col_mask = col_vals > chainA_length
        inter = self.raw_couplings.loc[row_mask, col_mask].copy()
        inter.columns = (col_vals[col_mask].to_numpy(dtype=int) - int(chainA_length))
        return inter

    def FilterMaskedBySavedColumns(self, saved_columns, source_df):
        if len(saved_columns) < 2:
            raise RuntimeError('Saved columns must contain at least two per-protein lists')

        saved_a = pd.to_numeric(pd.Index(saved_columns[0]), errors='coerce')
        saved_b = pd.to_numeric(pd.Index(saved_columns[1]), errors='coerce')
        if saved_a.isna().any() or saved_b.isna().any():
            raise RuntimeError('Saved columns must be numeric to restore residue numbering')

        row_labels = (saved_a.to_numpy(dtype=int) + 1).tolist()
        col_labels = (saved_b.to_numpy(dtype=int) + 1).tolist()

        row_labels = [label for label in row_labels if label in source_df.index]
        col_labels = [label for label in col_labels if label in source_df.columns]

        inter = source_df.loc[row_labels, col_labels].copy()
        inter.index = row_labels
        inter.columns = col_labels
        return inter

    def NormalizedCouplings(self, N_eff, full=False):
        """
        Convert the original coupling scores to EVcomplex scores.
        Source: Hopf et al. eLife 2014;3:e03430

        Parameters
        ----------
        `N_eff` — float
            The effective number of sequences in the MSA
        `full` — bool
            If True, return the full normalized matrix; otherwise, return only the interprotein couplings.

        Returns
        -------
        `pandas.DataFrame`
            The normalized coupling score matrix

        Raises
        ------
        `RuntimeError`
            If no interprotein coupling was found or the minimum coupling score is 0.
        """
        if full:
            if self.raw_couplings.empty:
                raise RuntimeError('No couplings found')
            else:
                df = self.raw_couplings.apply(pd.to_numeric, errors="coerce")
        else:
            if self.interprotein_couplings.empty:
                raise RuntimeError('No interprotein couplings found')
            else:
                df = self.interprotein_couplings.apply(pd.to_numeric, errors="coerce")
        if df.to_numpy().size == 0 or np.isnan(df.to_numpy()).all():
            raise RuntimeError('No couplings found')

        # The normaliser comes from the same matrix being normalised - the block
        # is self-normalised.  Scaling one block by another block's minimum is
        # not a self-consistent statistic.
        L = self.raw_couplings.shape[0]
        constant = 1 + (N_eff / L) ** -0.5
        abs_min = abs(np.nanmin(df.to_numpy()))
        if abs_min == 0:
            raise RuntimeError('Cannot normalize: minimum absolute score is 0')

        return (df / abs_min) / constant

    def SumUpNormalizedCouplings(self, N_eff, full=False):
        return float(np.nansum(self.NormalizedCouplings(N_eff, full).to_numpy()))

    def MaxNormalizedCouplings(self, N_eff, full=False):
        return float(np.nanmax(self.NormalizedCouplings(N_eff, full).to_numpy()))

    def GetBenchmarkIndex(self, ECfile_query):
        if type(self.interface) == None or not self.couplings_baseline: raise RuntimeError('No interface or couplings found')

        couplings_query = self.GetInterProtCouplingsByChainALength(ECfile_query, self.chain1_length, self.EC_threshold)

        TP_Query = self.TruePositiveCount(self.interface, couplings_query)
        if TP_Query == 0: raise RuntimeError(f'No true positive found for {ECfile_query}')
        TPRfull = self.TruePositiveRatio(self.TP_Baseline, TP_Query)
        TPR = self.TruePositiveRate(TP_Query, len(couplings_query))
        BI = self.BenchmarkIndex(TPRfull, TPR)
        print(f'TP Baseline: {self.TP_Baseline};\tTP Query: {TP_Query};\tTPRfull (%): {TPRfull:.2%};\tTPR (%): {TPR:.2%};\tBI: {BI:.3f}')
        return {'TP Baseline': self.TP_Baseline, 'TP Query': TP_Query, 'TPRfull': TPRfull, 'TPR': TPR, 'BI': BI}

    def SpearmanCorrelation(self, query_analyzer):
        """
        Return the Spearman correlation between the baseline and query couplings.

        Parameters
        ----------
        `query_analyzer` — ComplexAnalyzer
            The query analyzer instance to compare against.

        Returns
        -------
        Tuple[float, float]
            The Spearman correlation coefficient and p-value.
        """
        def _df_to_pair_dict(df):
            dfn = df.apply(pd.to_numeric, errors="coerce")
            stacked = dfn.stack(future_stack=True)
            return {(i, j): float(v) for (i, j), v in stacked.items() if np.isfinite(v)}

        ranked_baseline = dict(
            sorted(
                _df_to_pair_dict(self.interprotein_couplings).items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        )
        ranked_query = dict(
            sorted(
                _df_to_pair_dict(query_analyzer.interprotein_couplings).items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        )
        corr, pval = super().SpearmanCorrelation(ranked_baseline, ranked_query)
        return float(corr), float(pval)

    def EmpiricalDistribution(self):
        if self.interprotein_couplings.empty:
            raise RuntimeError('No interprotein couplings found')
        vals = pd.to_numeric(self.interprotein_couplings.values.ravel(), errors="coerce")
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            raise RuntimeError('No interprotein couplings found')
        res = super().EmpiricalDistribution(vals)
        return res

    def PlotCouplings(self):
        plt.figure(figsize=(10, 6))

        scores_fp = []  # False positives (red)
        distances_fp = []
        scores_tp = []  # True positives (green)
        distances_tp = []
        scores_other = []  # Others (black)
        distances_other = []

        dfn = self.interprotein_couplings.apply(pd.to_numeric, errors="coerce")
        stacked = dfn.stack(future_stack=True).astype(float).dropna()

        for (pos1, pos2), score in stacked.items():
            distance = 50.0
            if self.interface is not None:
                has_numeric_labels = pos1 in self.interface.index and pos2 in self.interface.columns
                has_string_labels = f'{pos1}' in self.interface.index and f'{pos2}' in self.interface.columns

                if has_numeric_labels:
                    distance = float(self.interface.loc[pos1, pos2])
                elif has_string_labels:
                    distance = float(self.interface.loc[f'{pos1}', f'{pos2}'])

            if score > self.EC_threshold and distance > self.distance_cutoff:
                scores_fp.append(score)
                distances_fp.append(distance)
            elif score > self.EC_threshold and distance < self.distance_cutoff:
                scores_tp.append(score)
                distances_tp.append(distance)
            else:
                scores_other.append(score)
                distances_other.append(distance)

        plt.scatter(scores_other, distances_other, marker='.', alpha=0.7, color='black', label='Negatives')
        plt.scatter(scores_fp, distances_fp, marker='.', alpha=0.7, color='red', label='False Positives')
        plt.scatter(scores_tp, distances_tp, marker='.', alpha=0.7, color='green', label='True Positives')
        plt.axhline(y=self.distance_cutoff, color='r', linestyle='--', label=f'Distance cutoff ({self.distance_cutoff} Å)')
        plt.axvline(x=self.EC_threshold, color='g', linestyle='--', label=f'EC threshold ({self.EC_threshold})')

        plt.xlabel('Evolutionary Coupling Score')
        plt.ylabel('Distance (Å)')
        plt.title('Evolutionary Coupling Score vs. Residue Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set fixed x-axis range from -0.1 to 1.8
        plt.xlim(-0.1, 5)

        return plt.gcf()

    def TopKPairs(self, k):
        """
        Get the top-k interprotein coupling pairs based on their scores.

        Parameters
        ----------
        `k` — int
            The number of top interprotein coupling pairs to retrieve.

        Returns
        -------
        `pd.DataFrame`
            A DataFrame containing the top-k interprotein coupling pairs, where each row is a pair (pos1, pos2) and the column represents the coupling score.

        Raises
        ------
        `ValueError`
            If `k` is not a positive integer.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        if self.interprotein_couplings.empty:
            return pd.DataFrame()

        dfn = self.interprotein_couplings.apply(pd.to_numeric, errors="coerce")
        stacked = dfn.stack(future_stack=True).astype(float).dropna()
        if stacked.empty:
            return pd.DataFrame()
        
        # Select top-k by score (descending)
        top = stacked.nlargest(k)
        return top

    def TopKInterproteinPairOverlap(self, query_analyzer, k):
        """
        Count how many top-k interprotein coupling pairs are shared with query_analyzer.

        Parameters
        ----------
        `query_analyzer` — ComplexAnalyzer
            The query analyzer instance to compare against.
        `k` — int
            The number of top interprotein coupling pairs to consider.

        Returns
        -------
        `int`
            The number of shared top-k interprotein coupling pairs.

        Raises
        ------
        `TypeError`
            If `query_analyzer` is not a ComplexAnalyzer instance.
        `ValueError`
            If `k` is not a positive integer.
        """
        if query_analyzer is None:
            raise TypeError('query_analyzer must be a ComplexAnalyzer instance')
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        if self.interprotein_couplings.empty or query_analyzer.interprotein_couplings.empty:
            return 0

        top_self = set(self.TopKPairs(k).index.tolist())             # (row_index, col_index)
        top_query = set(query_analyzer.TopKPairs(k).index.tolist())  # (row_index, col_index)
        return int(len(top_self.intersection(top_query)))

    def TopKTruePositiveCount(self, k):
        """
        Count the number of true positives among the top-k interprotein coupling pairs.

        Parameters
        ----------
        `k` — int
            The number of top interprotein coupling pairs to consider.
        
        Returns
        -------
        `int`
            The number of true positives among the top-k interprotein coupling pairs.

        Raises
        ------
        `ValueError`
            If `k` is not a positive integer.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        if self.interprotein_couplings.empty:
            return 0

        top_k = self.TopKPairs(k)
        if top_k.empty:
            return 0
        return self.TruePositiveCount(self.interface, top_k.to_dict())

    def TopKTruePositiveRate(self, k):
        """
        Calculate the true positive rate among the top-k interprotein coupling pairs.

        Parameters
        ----------
        `k` — int
            The number of top interprotein coupling pairs to consider.
        
        Returns
        -------
        `float`
            The true positive rate among the top-k interprotein coupling pairs.

        Raises
        ------
        `ValueError`
            If `k` is not a positive integer.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        if self.interprotein_couplings.empty:
            return 0.0

        top_k = self.TopKPairs(k)
        if top_k.empty:
            return 0.0
        tp_count = self.TruePositiveCount(self.interface, top_k.to_dict())

        return tp_count / k


class BaseCalibrator(object):
    """Maps a pair's coupling matrix into a calibrated interaction probability.

    ``RANKER``
        Selected to maximise AUPRC at the benchmark prevalence.  Scores EVERY
        pair on a graded scale, which is what `log_lr` and `pcontact` need.
    ``DETECTOR``
        Selected to maximise AUPRC at `PI`, i.e. tuned on the extreme head of the
        ranking where a proteome-scale screen actually operates.  Decides
        `high_precision_hit` and nothing else.
    """

    #: assumed proteome-wide interaction prevalence
    PI = 1e-3
    #: logit of the 0.12343 training prevalence the slopes were fitted at
    LOGIT_TAU = -1.9603207131817926
    LOGIT_PI = -6.906754778648554
    #: log(3984 labelled negatives) - the finest likelihood ratio they can show
    MAX_LOG_LR = 8.290041618704489
    LOW_CUT = 0.8734290196703057

    RANKER = None
    DETECTOR = None

    def __init__(self, pi=None):
        if pi is not None:
            self.PI = float(pi)
            self.LOGIT_PI = float(np.log(self.PI / (1.0 - self.PI)))

    # -- feature helpers ------------------------------------------------------
    @staticmethod
    def _normalized_entropy(counts, k):
        """Shannon entropy of `counts` over log(k), so it does not track size.

        Normalising by log(k) rather than by the number of residues is what makes
        this size-invariant: k top pairs can touch at most k distinct residues,
        however long the proteins are.
        """
        p = counts[counts > 0].astype(np.float64)
        if p.size == 0 or k <= 1:
            return 0.0
        p = p / p.sum()
        return float(-(p * np.log(p)).sum() / np.log(k))

    def _feature(self, S, flat, name):
        if name == "mean_cn":
            return float(flat.mean())
        if name.startswith("frac_above_"):
            t = float(name[len("frac_above_"):].replace("p", "."))
            return float((flat >= t).sum()) / flat.size
        if name == "max_cn":
            return float(flat.max())
        if name == "max_cn_sq":
            peak = float(flat.max())
            return float(np.sign(peak) * peak * peak)
        if name == "max_cn_log":
            return float(np.log1p(max(float(flat.max()), 0.0)))
        if name.startswith("conc_entropy_"):
            k = min(int(name[len("conc_entropy_"):]), flat.size)
            idx = np.argpartition(flat, flat.size - k)[flat.size - k:]
            rows, cols = np.unravel_index(idx, S.shape)
            return 0.5 * (self._normalized_entropy(np.bincount(rows), k)
                          + self._normalized_entropy(np.bincount(cols), k))
        raise KeyError(f"unknown calibration feature {name!r}")

    def _model_log_lr(self, analyzer, model, N_eff):
        """Evidence (log likelihood ratio) one model assigns to this pair."""
        S = analyzer.NormalizedCouplings(N_eff).to_numpy(dtype=float)
        flat = S.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            raise ValueError("normalised matrix has no finite entries")
        x = np.array([self._feature(S, flat, n) for n in model["features"]])
        x = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
        return float(x @ np.asarray(model["coef"]) + model["b0"] - self.LOGIT_TAU)

    # -- public API -----------------------------------------------------------
    def ConfidenceFlag(self, coverage):
        """Reliability flag from coverage-per-length (M_eff / L): `low` or `high`.

        A DATA-QUALITY judgement - is the alignment deep enough for DCA to say
        anything - and independent of how high the pair scores.
        """
        return False if coverage < self.LOW_CUT else True

    def Calibrate(self, analyzer, N_eff):
        """Score one pair.

        Parameters
        ----------
        `analyzer` — ComplexAnalyzer
            Built on the pair's coupling matrix, so that `interprotein_couplings`
            holds the inter-protein block restricted to the retained columns.
        `N_eff` — float
            Effective (reweighted) sequence count of the concatenated alignment.

        Returns
        -------
        dict with `log_lr_raw`, `pcontact` (that evidence as a probability at PI),
        `confidence_flag` (`False` for low /`True` for high) and `high_precision_hit`.
        """
        if self.RANKER is None or self.DETECTOR is None:
            raise NotImplementedError(
                f"{type(self).__name__} defines no RANKER/DETECTOR constants")
        L = int(analyzer.raw_couplings.shape[0])
        log_lr_raw = self._model_log_lr(analyzer, self.RANKER, N_eff)
        # logit(P at prior q) = log_lr + logit(q), so log_lr is the prevalence-free part
        # Capped because with 3984 labelled negatives no likelihood ratio finer than 1/3984 can be demonstrated
        log_lr = min(log_lr_raw, self.MAX_LOG_LR)
        hit_log_lr = self._model_log_lr(analyzer, self.DETECTOR, N_eff)
        coverage = float(N_eff) / float(L)
        return {
            "log_lr_raw": log_lr_raw,
            "pcontact": float(1.0 / (1.0 + np.exp(-(log_lr + self.LOGIT_PI)))),
            "confidence_flag": self.ConfidenceFlag(coverage),
            "high_precision_hit": bool(hit_log_lr >= self.DETECTOR["hit_log_lr"]),
            "neffoverL": coverage,
        }


class BuiltinPLMCalibrator(BaseCalibrator):
    """Calibration fitted on plmDCA couplings.

    Held out: ranker AUC 0.879, AUPRC 0.803, ECE 0.040.
    """

    LOW_CUT = 0.8734290196703057

    RANKER = {
        "features": ("frac_above_0", "max_cn_sq", "conc_entropy_10"),
        "coef": (1.261770011395687, 4.421236943580058, -0.44171569920052084),
        "mean": (0.1871615664329687, 0.7600967167561608, 0.9547690956719593),
        "scale": (0.08127383738222799, 2.416452373788886, 0.04797641421035466),
        "b0": -2.4838705529482583,
    }
    DETECTOR = {
        "features": ("frac_above_1p25",),
        "coef": (49.21454021348768,),
        "mean": (1.0326145352227491e-05,),
        "scale": (0.0001812544100640886,),
        "b0": 0.5211322518556427,
        # held-out 0.1%-FPR operating point; fires on 159 positives / 2 negatives
        "hit_log_lr": 1.4813428297931597,
    }


class BuiltinGaussCalibrator(BaseCalibrator):
    """Calibration fitted on GaussDCA couplings.

    Held out: ranker AUC 0.890, AUPRC 0.801, ECE 0.033.
    """

    LOW_CUT = 1.5964534737486649

    RANKER = {
        "features": ("frac_above_0", "max_cn_sq"),
        "coef": (1.58675901061756, 1.39192051940705),
        "mean": (0.16203104713451555, 0.7566148060584378),
        "scale": (0.09029099655236972, 2.4423143207822986),
        "b0": -2.9314009156717273,
    }
    DETECTOR = {
        "features": ("frac_above_0p001", "max_cn_log"),
        "coef": (1.517195400105569, 0.5592752338008525),
        "mean": (0.1603621965835015, 0.5276021495776804),
        "scale": (0.08956596002756717, 0.18479331879786745),
        "b0": -3.0370236622651228,
        # held-out 0.1%-FPR operating point; fires on 148 positives / 5 negatives
        "hit_log_lr": 3.1020144107309235,
    }
