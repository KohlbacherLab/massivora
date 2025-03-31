import logging

import pandas as pd
from ost import io
from ost.geom import MinDistance, Vec3List

logger = logging.getLogger(__name__)

class BaseBenchmarker(object):
    def __init__(self, EC_threshold=0.7, distance_cutoff=8):
        self.distance_cutoff = distance_cutoff
        self.EC_threshold = EC_threshold

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

    def AnalyzeCouplings(self, ECfile, threshold=0.7):
        with open(ECfile, 'r') as f:
            lines = f.readlines()
            data = {}
            for line in lines[1:]:
                # parts = line.strip().split()
                # if len(parts) == 6:
                #     pos1, aa1, pos2, aa2, _, score = parts
                parts = line.strip().split(',')
                pos1, pos2, score = int(parts[0]), int(parts[1]), float(parts[2])
                data[pos1, pos2] = score
        positives = {k: v for k, v in data.items() if v > threshold}
        if not positives: return None
        return positives

    def TruePositiveCount(self, interface, couplings):
        TP = 0
        for pos1, pos2 in couplings.keys():
            if f'{pos1}' in interface.index and f'{pos2}' in interface.columns:
                if interface.loc[f'{pos1}', f'{pos2}'] < self.distance_cutoff:
                    print(f'Found true positive: {pos1}, {pos2} ({interface.loc[f"{pos1}", f"{pos2}"]})')
                    TP += 1
        return TP

    def TruePositiveRatio(self, baseline_cp, new_cp):
        return new_cp / baseline_cp

    def TruePositiveRate(self, nTP, nCouplings):
        return nTP / nCouplings

    def BenchmarkIndex(self, TPRfull, TPR):
        return pow(TPR,0.8)*pow(TPRfull, 0.2)

    def GetBenchmarkIndex(self, ECfile_query):
        if type(self.interface) == None or not self.couplings_baseline: raise RuntimeError('No interface or couplings found')

        couplings_query = self.AnalyzeCouplings(ECfile_query, self.EC_threshold)

        TP_Query = self.TruePositiveCount(self.interface, couplings_query)
        TPRfull = self.TruePositiveRatio(self.TP_Baseline, TP_Query)
        TPR = self.TruePositiveRate(TP_Query, len(couplings_query))
        BI = self.BenchmarkIndex(TPRfull, TPR)
        print(f'TP Baseline: {self.TP_Baseline};\tTP Query: {TP_Query};\tTPRfull (%): {TPRfull:.2%};\tTPR (%): {TPR:.2%};\tBI: {BI:.3f}')
        return {'TP Baseline': self.TP_Baseline, 'TP Query': TP_Query, 'TPRfull': TPRfull, 'TPR': TPR, 'BI': BI}

class MonomerBenchmarker(BaseBenchmarker):
    def __init__(self, structure_file, chain_name, ECfile_baseline, EC_threshold=0.7, distance_cutoff=8):
        super().__init__(EC_threshold, distance_cutoff)
        self.mol = io.LoadEntity(structure_file).Select('peptide=true')
        # Extract the chains
        chain_handle = self.mol.FindChain(chain_name)
        # Analyze the interface and the couplings
        self.interface = self.AnalyzeProteinInterface(chain_handle, chain_handle, self.distance_cutoff)
        self.couplings_baseline = self.AnalyzeCouplings(ECfile_baseline, self.EC_threshold)
        if not self.couplings_baseline: raise RuntimeError(f'No couplings found under the threshold {self.EC_threshold}')
        self.TP_Baseline = self.TruePositiveCount(self.interface, self.couplings_baseline)
        if self.TP_Baseline == 0: raise RuntimeError(f'No true positive found for {ECfile_baseline}')


class ComplexBenchmarker(BaseBenchmarker):
    def __init__(self, structure_file, chain1_name, chain1_length, chain2_name, ECfile_baseline, EC_threshold=0.7, distance_cutoff=8):
        super().__init__(EC_threshold, distance_cutoff)
        self.chain1_length = chain1_length
        self.mol = io.LoadEntity(structure_file).Select('peptide=true')
        # Extract the chains
        chain1_handle = self.mol.FindChain(chain1_name)
        chain2_handle = self.mol.FindChain(chain2_name)
        # Analyze the interface and the couplings
        self.interface = self.AnalyzeProteinInterface(chain1_handle, chain2_handle, self.distance_cutoff)
        self.couplings_baseline = self.AnalyzeCouplings(ECfile_baseline, chain1_length, self.EC_threshold)
        if not self.couplings_baseline: raise RuntimeError(f'No couplings found under the threshold {self.EC_threshold}')
        self.TP_Baseline = self.TruePositiveCount(self.interface, self.couplings_baseline)
        if self.TP_Baseline == 0: raise RuntimeError(f'No true positive found for {ECfile_baseline}')

    def AnalyzeCouplings(self, ECfile, chain1_length, threshold=0.7):
        with open(ECfile, 'r') as f:
            lines = f.readlines()
            data = {}
            for line in lines[1:]:
                parts = line.strip().split(',')
                pos1, pos2, score = int(parts[0]), int(parts[1]), float(parts[2])
                if pos2 < chain1_length or pos1 > chain1_length:
                    continue
                if pos1 > chain1_length:
                    pos1 = pos1 - chain1_length
                if pos2 > chain1_length:
                    pos2 = pos2 - chain1_length
                data[pos1, pos2] = score
        positives = {k: v for k, v in data.items() if v > threshold}
        if not positives: return None
        return positives

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

if __name__ == '__main__':
    # bm = MonomerBenchmarker('/Users/simon/research/EVcouplings/MSA_subset_evaluation/1g79.cif', 'A', '/Users/simon/research/EVcouplings/MSA_subset_evaluation/coupling/PDXH_ECOLI_1-218_b0.5_Full_ECs.txt')
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

    bm = ComplexBenchmarker('/Users/simon/Downloads/6wlz.cif', 'D', 511, 'A', '/Users/simon/Dev/EVsnap/GaussDCA_ATPA_HUMAN-ATPB_HUMAN_b0.2.csv', 1)
    # bm.GetBenchmarkIndex('/Users/simon/Dev/EVsnap/GaussDCA_ATPA_HUMAN-ATPB_HUMAN_b0.2.csv')