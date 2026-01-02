import logging
import zarr
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ecdf
from ost import io
from ost.geom import MinDistance, Vec3List

logger = logging.getLogger(__name__)

class BaseBenchmarker(object):
    def __init__(self, EC_threshold=0.7, distance_cutoff=8, min_seq_length=1):
        self.distance_cutoff = distance_cutoff
        self.EC_threshold = EC_threshold
        self.min_seq_length = min_seq_length

    def AnalyzeProteinInterface(self, chainAhandle, chainBhandle, cutoff=8):
        if not (chainAhandle.valid and chainBhandle.valid): return None
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
        if not (surfaceAlist and surfaceBlist): return None
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
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                df.iloc[j, i] = df.iloc[i, j]
        return df

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
        df = pd.DataFrame(0, index=positions, columns=positions, dtype=float)
        for _, row in long_df.iterrows():
            df.at[row["res1"], row["res2"]] = row["score"]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                df.iloc[j, i] = df.iloc[i, j]
        return df

    def ParseNumpyOutput(self, ECfile):
        arr = np.load(ECfile, allow_pickle=True)
        data = {}
        for i in range(arr.shape[0]):
            for j in range(i+1, arr.shape[1]):
                data[(i+1, j+1)] = arr[i, j]
        return data

    def ParseZarrOutput(self, ECfile):
        arr = zarr.load(ECfile)
        with open(os.path.join(ECfile, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        saved_columns = metadata.get('saved_columns', [])
        data = {}
        for i in range(arr.shape[0]):
            for j in range(i+1, arr.shape[1]):
                data[(saved_columns[i], saved_columns[j])] = arr[i, j]
        return data

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

class MonomerBenchmarker(BaseBenchmarker):
    def __init__(self, structure_file=None, chain_name=None, ECfile_baseline=None, EC_threshold=0.7, distance_cutoff=8, min_seq_length=1):
        super().__init__(EC_threshold, distance_cutoff, min_seq_length)
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
        else:
            self.raw_couplings = {}
        self.positive_couplings = self.AnalyzeCouplings()
        if not self.positive_couplings: 
            self.positive_couplings = {}
            print(f'No couplings found under the threshold {self.EC_threshold}')

    def AnalyzeCouplings(self):
        positives = {}
        for pair, score in self.raw_couplings.items():
            pos1, pos2 = pair
            if score >= self.EC_threshold:
                if abs(pos1 - pos2) >= self.min_seq_length:
                    positives[(pos1, pos2)] = score
        if not positives: return None
        return positives

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
    
    def SpearmanCorrelation(self, query_benchmarker):
        """
        Return the Spearman correlation between the baseline and query couplings.

        Parameters
        ----------
        `query_benchmarker` — MonomerBenchmarker
            The query benchmarker instance to compare against.

        Returns
        -------
        Tuple[float, float]
            The Spearman correlation coefficient and p-value.
        """
        if not query_benchmarker:
            return 0, 0
        ranked_baseline = dict(sorted(self.raw_couplings.items(), key=lambda item: (item[0][0], item[0][1])))
        ranked_query = dict(sorted(query_benchmarker.raw_couplings.items(), key=lambda item: (item[0][0], item[0][1])))
        corr, pval = super().SpearmanCorrelation(ranked_baseline, ranked_query)
        return float(corr), float(pval)

    def EmpiricalDistribution(self):
        scores = list(self.raw_couplings.values())
        res = super().EmpiricalDistribution(scores)
        return res

    def GetBenchmarkIndex(self, query_benchmarker):
        if type(self.interface) == None or not self.positive_couplings: raise RuntimeError('No interface or couplings found')
        if not isinstance(query_benchmarker, MonomerBenchmarker):
            raise TypeError('The input must be an instance of MonomerBenchmarker')

        couplings_query = query_benchmarker.positive_couplings

        nTP_Query = super().TruePositiveCount(self.interface, couplings_query)
        TPRfull = self.TruePositiveRatio(self.nTP, nTP_Query)
        TPR = super().TruePositiveRate(nTP_Query, len(couplings_query))
        BI = self.BenchmarkIndex(TPRfull, TPR)
        Spearman = self.SpearmanCorrelation(query_benchmarker)
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

class ComplexBenchmarker(BaseBenchmarker):
    def __init__(self, structure_file=None, chain1_name="A", chain1_length=None, chain2_name=None, ECfile_baseline=None, EC_threshold=0.7, distance_cutoff=8):
        super().__init__(EC_threshold, distance_cutoff)
        self.nTP = None
        self.positive_couplings = pd.DataFrame()
        if structure_file is not None:
            self.ReadStructure(structure_file, chain1_name, chain2_name)
        if ECfile_baseline is not None:
            self.ReadCouplings(ECfile_baseline, chain1_length)

    def ReadStructure(self, structure_file, chain1_name, chain2_name):
        self.mol = io.LoadEntity(structure_file).Select('peptide=true')
        chain1_handle = self.mol.FindChain(chain1_name)
        chain2_handle = self.mol.FindChain(chain2_name)
        self.interface = self.AnalyzeProteinInterface(chain1_handle, chain2_handle, self.distance_cutoff)

    def ReadCouplings(self, ECfile_baseline, chain1_length):
        if ECfile_baseline.endswith('.csv'):
            self.raw_couplings = self.ParseCSVOutput(ECfile_baseline)
        elif ECfile_baseline.endswith('.txt'):
            self.raw_couplings = self.ParsePLMCOutput(ECfile_baseline)
        elif ECfile_baseline.endswith('.npy'):
            self.raw_couplings = self.ParseNumpyOutput(ECfile_baseline)
        else:
            self.raw_couplings = pd.DataFrame()
        # Analyze the interface and the couplings
        self.interprotein_couplings = self.AnalyzeCouplings(chain1_length)
        inter = self.interprotein_couplings.apply(pd.to_numeric, errors="coerce")
        self.positive_couplings = inter.where(inter >= self.EC_threshold)
        if self.positive_couplings.notna().sum().sum() == 0:
            self.positive_couplings = pd.DataFrame()
            print(f'No couplings found under the threshold {self.EC_threshold}')

    def SumUpRawCouplings(self):
        if self.raw_couplings.empty:
            raise RuntimeError('No raw couplings found')
        vals = pd.to_numeric(self.raw_couplings.values.ravel(), errors="coerce")
        if np.isnan(vals).all():
            raise RuntimeError('No raw couplings found')
        return float(np.nansum(vals))

    def SumUpNormalizedCouplings(self, N_eff, L):
        if self.raw_couplings.empty:
            raise RuntimeError('No raw couplings found')
        constant = 1 + (N_eff / L) ** -0.5
        vals = pd.to_numeric(self.raw_couplings.values.ravel(), errors="coerce")
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            raise RuntimeError('No raw couplings found')
        abs_min = abs(np.nanmin(vals))
        if abs_min == 0:
            raise RuntimeError('Cannot normalize: minimum absolute score is 0')
        return float(np.sum((vals / abs_min) / constant))

    def SumUpNormalizedInterProtCouplings(self, N_eff, L):
        if self.interprotein_couplings.empty:
            raise RuntimeError('No interprotein couplings found')
        constant = 1 + (N_eff / L) ** -0.5
        vals = pd.to_numeric(self.interprotein_couplings.values.ravel(), errors="coerce")
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            raise RuntimeError('No interprotein couplings found')
        abs_min = abs(np.nanmin(vals))
        if abs_min == 0:
            raise RuntimeError('Cannot normalize: minimum absolute score is 0')
        return float(np.sum((vals / abs_min) / constant))

    def MaxNormalizedInterProtCouplings(self, N_eff, L):
        if self.interprotein_couplings.empty:
            raise RuntimeError('No interprotein couplings found')
        constant = 1 + (N_eff / L) ** -0.5
        vals = pd.to_numeric(self.interprotein_couplings.values.ravel(), errors="coerce")
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            raise RuntimeError('No interprotein couplings found')
        abs_min = abs(np.nanmin(vals))
        if abs_min == 0:
            raise RuntimeError('Cannot normalize: minimum absolute score is 0')
        return float(np.nanmax((vals / abs_min) / constant))

    def AnalyzeCouplings(self, saved_columns):
        if type(saved_columns) == int:
            inter = self.raw_couplings.loc[:saved_columns, saved_columns:]
            inter.columns -= saved_columns
        else:
            if len(saved_columns) != len(self.raw_couplings.index):
                raise RuntimeError('Saved columns length does not match raw couplings length')
            diff = np.diff(saved_columns)
            prot2_start = np.where(diff < 0)[0][0]+1
            inter = self.raw_couplings.iloc[:prot2_start, prot2_start:]
            inter.index = saved_columns[:prot2_start]
            inter.columns = saved_columns[prot2_start:]
            inter.index += 1
            inter.columns += 1
        return inter

    def GetBenchmarkIndex(self, ECfile_query):
        if type(self.interface) == None or not self.couplings_baseline: raise RuntimeError('No interface or couplings found')

        couplings_query = self.AnalyzeCouplings(ECfile_query, self.chain1_length, self.EC_threshold)

        TP_Query = self.TruePositiveCount(self.interface, couplings_query)
        if TP_Query == 0: raise RuntimeError(f'No true positive found for {ECfile_query}')
        TPRfull = self.TruePositiveRatio(self.TP_Baseline, TP_Query)
        TPR = self.TruePositiveRate(TP_Query, len(couplings_query))
        BI = self.BenchmarkIndex(TPRfull, TPR)
        print(f'TP Baseline: {self.TP_Baseline};\tTP Query: {TP_Query};\tTPRfull (%): {TPRfull:.2%};\tTPR (%): {TPR:.2%};\tBI: {BI:.3f}')
        return {'TP Baseline': self.TP_Baseline, 'TP Query': TP_Query, 'TPRfull': TPRfull, 'TPR': TPR, 'BI': BI}

    def SpearmanCorrelation(self, query_benchmarker):
        """
        Return the Spearman correlation between the baseline and query couplings.

        Parameters
        ----------
        `query_benchmarker` — ComplexBenchmarker
            The query benchmarker instance to compare against.

        Returns
        -------
        Tuple[float, float]
            The Spearman correlation coefficient and p-value.
        """
        def _df_to_pair_dict(df):
            dfn = df.apply(pd.to_numeric, errors="coerce")
            stacked = dfn.stack(future_stack=True)
            return {(i, j): float(v) for (i, j), v in stacked.items()}

        ranked_baseline = dict(
            sorted(
                _df_to_pair_dict(self.interprotein_couplings).items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        )
        ranked_query = dict(
            sorted(
                _df_to_pair_dict(query_benchmarker.interprotein_couplings).items(),
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

        for pair, score in self.interprotein_couplings.items():
            pos1, pos2 = pair
            distance = 50
            if self.interface is not None:
                if f'{pos1}' in self.interface.index and f'{pos2}' in self.interface.columns:
                    distance = self.interface.loc[f'{pos1}', f'{pos2}']
                    if score > self.EC_threshold and distance > self.distance_cutoff:
                        scores_fp.append(score)
                        distances_fp.append(distance)
                    elif score > self.EC_threshold and distance < self.distance_cutoff:
                        scores_tp.append(score)
                        distances_tp.append(distance)
                else:
                    scores_other.append(score)
                    distances_other.append(distance)
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

    def TopKInterproteinPairOverlap(self, query_benchmarker, k):
        """
        Count how many top-k interprotein coupling pairs are shared with query_benchmarker.

        Parameters
        ----------
        `query_benchmarker` — ComplexBenchmarker
            The query benchmarker instance to compare against.
        `k` — int
            The number of top interprotein coupling pairs to consider.

        Returns
        -------
        `int`
            The number of shared top-k interprotein coupling pairs.

        Raises
        ------
        `TypeError`
            If `query_benchmarker` is not a ComplexBenchmarker instance.
        `ValueError`
            If `k` is not a positive integer.
        """
        if query_benchmarker is None:
            raise TypeError('query_benchmarker must be a ComplexBenchmarker instance')
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        if self.interprotein_couplings.empty or query_benchmarker.interprotein_couplings.empty:
            return 0

        def _topk_pairs(df, k_):
            dfn = df.apply(pd.to_numeric, errors="coerce")
            stacked = dfn.stack(future_stack=True)
            if stacked.empty:
                return set()
            # Select top-k by score (descending)
            top = stacked.nlargest(k_)
            return set(top.index.tolist())  # (row_index, col_index)

        top_self = _topk_pairs(self.interprotein_couplings, k)
        top_query = _topk_pairs(query_benchmarker.interprotein_couplings, k)
        return int(len(top_self.intersection(top_query)))

if __name__ == '__main__':
    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/MSA_subset_evaluation/coupling/PDXH_ECOLI_1-218_b0.5_Full_ECs.txt', min_seq_length=5)
    # fig = bm.PlotCouplings()
    # fig.savefig('/Users/simon/Dev/CoevoFlash/1g79_couplings_plmc.png', dpi=300, bbox_inches='tight')
    # print("PLMC:", bm.TruePositiveCount(), f"{bm.TruePositiveRate():.2%}")
    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/Dev/CoevoFlash/new.csv', min_seq_length=5)
    # fig = bm.PlotCouplings()
    # fig.savefig('/Users/simon/Dev/CoevoFlash/1g79_couplings_aplm.png', dpi=300, bbox_inches='tight')
    # print("AsymJulia:", bm.TruePositiveCount(), f"{bm.TruePositiveRate():.2%}")
    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/Algorithm benchmark/Intra chain/GaussDCA_PDXH_ECOLI.csv', min_seq_length=5)
    # fig = bm.PlotCouplings()
    # fig.savefig('/Users/simon/Dev/CoevoFlash/1g79_couplings_gauss.png', dpi=300, bbox_inches='tight')
    # print("GaussDCA:", bm.TruePositiveCount(), f"{bm.TruePositiveRate():.2%}")
    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/Dev/CoevoFlash/Jmat_PDXH_ECOLI.npy', min_seq_length=5)
    # fig = bm.PlotCouplings()
    # print("AsymC++:", bm.TruePositiveCount(), f"{bm.TruePositiveRate():.2%}")
    # fig.savefig('/Users/simon/Dev/CoevoFlash/1g79_couplings_new.png', dpi=300, bbox_inches='tight')
    # results = []
    # for i in range(1, 5):
    #     result = bm.GetBenchmarkIndex(f'/Users/simon/research/EVcouplings/MSA_subset_evaluation/multi-iteration/it{i}/couplings/PDXH_ECOLI_1-218_b0.5_it{i}_ECs.txt')
    #     results.append(result)
    #     result = bm.GetBenchmarkIndex(f'/Users/simon/research/EVcouplings/MSA_subset_evaluation/multi-iteration/it{i}/couplings/PDXH_ECOLI_1-218_b0.5_it{i}_10000_ECs.txt')
    #     results.append(result)
    #     result = bm.GetBenchmarkIndex(f'/Users/simon/research/EVcouplings/MSA_subset_evaluation/multi-iteration/it{i}/couplings/PDXH_ECOLI_1-218_b0.5_it{i}_5000_ECs.txt')
    #     results.append(result)
    # df = pd.DataFrame(results)
    # df.to_csv('/Users/simon/research/EVcouplings/MSA_subset_evaluation/benchmark_results2.csv', index=False)

    # bm = ComplexBenchmarker(structure_file='/Users/simon/research/EVcouplings/E.coli-50S/6pj6-50S.cif', chain1_name='K', chain1_length=123, chain2_name='U', ECfile_baseline='/Users/simon/research/EVcouplings/E.coli-50S/couplings/plmc/RL14_ECOLI_RL24_ECOLI_b0.2.txt', EC_threshold=0.1)
    # print(sum(bm.interprotein_couplings.values()))
    # fig = bm.PlotCouplings()
    # fig.savefig('/Users/simon/Dev/CoevoFlash/6pj6-50S_couplings_plmc.png', dpi=300, bbox_inches='tight')

    # bm = ComplexBenchmarker('/Users/simon/Downloads/6wlz.cif', 'A', 617, 'E', '/Users/simon/Dev/CoevoFlash/Jmat_PDXH_ECOLI.npy', 0.1)
    # bm.GetBenchmarkIndex('/Users/simon/Dev/EVsnap/GaussDCA_ATPA_HUMAN-ATPB_HUMAN_b0.2.csv')

    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/Algorithm benchmark/Intra chain/GSH1_ECOLI_1-518_b0.2_old.txt', min_seq_length=5)
    # bm1 = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/Algorithm benchmark/Intra chain/GSH1_ECOLI_1-518_b0.2_new.txt', min_seq_length=5)
    # print(bm.SpearmanCorrelation(bm1))
    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/Algorithm benchmark/Intra chain/NUOG_ECOLI_1-908_b0.2_old.txt', min_seq_length=5)
    # bm1 = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/Algorithm benchmark/Intra chain/NUOG_ECOLI_1-908_b0.2_new.txt', min_seq_length=5)
    # print(bm.SpearmanCorrelation(bm1))
    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/Algorithm benchmark/Intra chain/PDXH_ECOLI_1-218_b0.2_old.txt', min_seq_length=5)
    # bm1 = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/Algorithm benchmark/Intra chain/PDXH_ECOLI_1-218_b0.2_new.txt', min_seq_length=5)
    # print(bm.SpearmanCorrelation(bm1))
    bm = ComplexBenchmarker(chain1_length=110, ECfile_baseline='/Users/simon/research/EVcouplings/E.coli-50S/couplings/plmc/RL22_ECOLI_RL31_ECOLI_b0.2.txt')
    print(bm.interprotein_couplings.values.max())
    with open("/Users/simon/Downloads/merged/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    bm1 = ComplexBenchmarker(chain1_length=metadata['saved_columns'], ECfile_baseline='/Users/simon/Downloads/newtest_zarr.txt')
    print(bm1.interprotein_couplings.values.max())
    print(bm.SpearmanCorrelation(bm1))
    print(bm1.TopKInterproteinPairOverlap(bm, 10))
    edf = bm1.EmpiricalDistribution()
    print(edf.cdf)
    print(edf.cdf.quantiles)
    print(edf.cdf.quantiles[np.where(edf.cdf.probabilities >= 0.9999)[0]])
    print(len(edf.cdf.quantiles[np.where(edf.cdf.probabilities >= 0.9999)[0]]))
    print(edf.cdf.confidence_interval(0.99).high.quantiles)