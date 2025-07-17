import os
import re
import time
import logging
import sys
import compute
import zarr
import pickle
import subprocess

import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO
from collections import OrderedDict
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio import ExPASy
from Bio import SwissProt
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)

class TextAlignment(MultipleSeqAlignment):
    def __init__(self, records):
        # Allowing contruction from file path
        if type(records) == str:
            if os.path.isfile(records):
                logger.info(f'Loading alignment from file: {records}')
                _, ext = os.path.splitext(records)
                self.alignmentName = os.path.basename(_)
                if ext.lower() == ".sto":
                    fmt = "stockholm"
                elif ext.lower().startswith(".fa") or ext.lower() == ".a2m":
                    fmt = "fasta"
                else:
                    logger.error(f'Input file {os.path.basename(_)} has an unsupported format {ext}')
                    raise NotImplementedError
                records = AlignIO.read(records, fmt)
            else: raise FileNotFoundError(f"File not found: {records}")
        elif isinstance(records, SeqRecord):
            # construct with single record
            records = [records]
        # construct with biopython supported iterable
        super().__init__(records)
        try:
            self._update_info()
        except IndexError:
            logger.error("Alignment is empty")
            raise ValueError("Alignment is empty")
        logger.debug(f'Alignment loaded: {len(self)} sequences, {len(self[0])} positions')

    def _parse_description(self, description):
        pattern = r"OS=(.*?)(?=\s+(?:GN|PE|SV)=|$)(?:\s(?:GN)=(.*?))?(?=\s+(?:PE|SV)=|$)(?:\s(?:PE)=(.*?))?(?=\s+(?:SV)=|$)(?:\s(?:SV)=(.*?))?$"
        match = re.search(pattern, description)
        if match:
            # matching pattern OS=... GN=... PE=... SV=...
            return match.group(1), match.group(2), match.group(3), match.group(4)
        return None, None, None, None

    def _update_info(self):
        self.querySequence = self._records[0].seq
        self.sequenceLength = len(self._records[0].seq)
        self.sequenceCount = len(self._records)

    @classmethod
    def From_SwissProt(cls, protein_id):
        try:
            handle = ExPASy.get_sprot_raw(protein_id)
            record = SeqIO.read(handle, 'swiss')
            return cls(record)
        except ValueError as e:
            logger.error(f"Failed to fetch protein {protein_id}: {e}")
            return None

    def AnalogueSearch(self, query=None, threshold=0.2, **search_args):
        if not query:
            _id = self._records[0].id
            query = os.path.join(search_args['working_dir'], f'{_id}.fasta')
            SeqIO.write(self._records[0], query, "fasta")
        # binary = '/Users/simon/Applications/hmmer-3.4/bin/jackhmmer'
        search_threshold = str(threshold * self.sequenceLength)
        cmd = [
            search_args['binary'],
            "-N", str(search_args['iterations']),
            "-o", '/dev/null',
            "-A", os.path.join(search_args['working_dir'], f'{_id}.sto'),
            "--tblout", os.path.join(search_args['working_dir'], f'{_id}.tblout'),
            "--domtblout", os.path.join(search_args['working_dir'], f'{_id}.domtblout'),
            "--noali",
            "--notextw",
            "-T", search_threshold,
            "--domT", search_threshold,
            "--incT", search_threshold,
            "--incdomT", search_threshold,
            "--cpu", str(search_args['cpu']),
            query, search_args['database']
        ]
        logger.debug("Running Jackhmmer with the following command line: "+ ' '.join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            logger.error(f"Jackhmmer failed with return code {r.returncode}")
            raise RuntimeError(f"Jackhmmer failed with return code {r.returncode}")
        else:
            self.__init__(os.path.join(search_args['working_dir'], f'{_id}.sto'))

    def Get_Numpy_Array(self, mapped=False, string=None):
        if not string: string = ''.join([str(record.seq) for record in self])
        matrix = compute.getAlignmentInNumpy(string, self.sequenceLength, self.sequenceCount)
        if mapped:
            # First change all the '.' to '-'
            matrix[matrix == b'.'] = b'-'
            # Then map the characters to numbers
            # All the invalid characters are mapped to 21
            mapping = np.full(128, 21, dtype=np.int8)
            valid_chars = 'acdefghiklmnpqrstvwyACDEFGHIKLMNPQRSTVWY-'
            values = np.arange(-20, 21, dtype=np.int8)
            for char, value in zip(valid_chars, values): mapping[ord(char)] = value
            ascii_codes = np.frombuffer(matrix.tobytes(), dtype=np.int8)
            mapped_flat = mapping[ascii_codes]
            matrix = mapped_flat.reshape(matrix.shape)
        return matrix

    def Filtering_MSA_Gap(self, row_keep_percentage, col_keep_percentage):
        # NOTE: this function is not going to filter out the invalid characters
        # The invalid characters are only removed when converting to zarr
        
        row_keep_percentage /= 100
        col_keep_percentage /= 100
        logger.debug("Calculate columns to keep")
        columns_to_keep = np.array([char != '-' for char in self._records[0].seq])
        start_time = time.time()
        matrix = self.Get_Numpy_Array()
        logger.debug(f"Time taken for numpylizing: {time.time() - start_time:.2f}s")
        matrix = matrix[:, columns_to_keep]
        # Calculate the ratio of '-' in each row and filter out rows with a ratio higher than 50%
        logger.debug("Calculate rows to keep")
        row_ratios = np.sum(matrix == b'-', axis=1) / matrix.shape[1]
        indices_to_keep = np.where(row_ratios <= row_keep_percentage)[0]
        indices_to_delete = np.where(row_ratios > row_keep_percentage)[0]
        matrix = matrix[indices_to_keep]
        logger.debug(f"Matrix shape after row filtering: {matrix.shape}")
        L = matrix.shape[0]
        logger.debug("Calculate columns to lowercase")
        col_ratios = np.sum(matrix == b'-', axis=0) / L
        cols_to_keep = np.where(col_ratios < col_keep_percentage)[0]
        self.saved_columns = cols_to_keep
        matrix = matrix[:, cols_to_keep]
        ############
        # The EVcouplings version: make the invalid columns lowercase and remove them in PLMC
        # col_indices_to_lowercase = np.where(col_ratios >= col_keep_percentage)[0]
        # for col_idx in col_indices_to_lowercase:
        #     matrix[:, col_idx] = np.char.lower(matrix[:, col_idx])
        ############
        logger.debug("Reconstructing alignment")
        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
        assert len(self._records) == L
        for i, record in enumerate(self):
            record.letter_annotations = {}
            byte_t = bytes()
            for j in matrix[i]: byte_t += j
            record.seq = Seq(byte_t.decode())

        self._update_info()

        logger.info(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Filtering_MSA_Gap2(self, row_keep_percentage, col_keep_percentage):
        ## This function is a bit more readable than the previous one, but it is one time slower
        # Get indices of gaps in first sequence
        gap_indices = [i for i, char in enumerate(self._records[0].seq) if char == '-']
        for record in self:
            seq_list = list(str(record.seq))
            for idx in sorted(gap_indices, reverse=True):
                seq_list.pop(idx)
            record.letter_annotations = {}
            record.seq = Seq(''.join(seq_list))

        # Remove sequences with too many gaps
        row_keep_percentage = int(round(row_keep_percentage / 100 * len(self._records[0].seq) + 0.5, 0))
        for i in range(len(self._records), -1, -1):
            # print(i,len(self._records) )
            gap_count = self._records[i-1].count('-')
            if gap_count >= row_keep_percentage:
                del self._records[i-1]

        col_keep_percentage = int(round(col_keep_percentage / 100 * len(self._records) + 0.5, 0))
        index_to_lowercase = []
        for i in range(len(self._records[0].seq)):
            gap_count = self[:, i].count('-')
            if gap_count >= col_keep_percentage:
                index_to_lowercase.append(i)
        for record in self:
            seq_list = list(str(record.seq))
            for idx in index_to_lowercase:
                seq_list[idx] = seq_list[idx].lower()
            record.seq = Seq(''.join(seq_list))

        self._update_info()

        logger.debug(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Filtering_MSA_Identity(self, threshold=0.8):
        # Calculate sequence identity for each sequence compared to the query sequence
        matrix = self.Get_Numpy_Array()
        query_seq = matrix[0]  # The first sequence is our reference
        identical_positions = np.equal(matrix, query_seq.reshape(1, -1))

        identity_scores = np.sum(identical_positions, axis=1) / self.sequenceLength
        indices_to_keep = np.where(identity_scores <= threshold)[0]
        if 0 not in indices_to_keep:
            indices_to_keep = np.insert(indices_to_keep, 0, 0)

        indices_to_delete = np.setdiff1d(np.arange(len(self)), indices_to_keep)
        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]

        self._update_info()
        logger.info(f'Filtered alignment contains: {len(self)} effective sequences')

    def __add__(self, other):
        if not isinstance(other, MultipleSeqAlignment):
            raise NotImplementedError("Only MultipleSeqAlignment can be concatenated")
        
        # Create species-based dictionary for both alignments
        self_species = {}
        other_species = {}

        # Group sequences by species in self alignment
        for rec in self[1:]:
            species,_,_,_ = self._parse_description(rec.description)
            if species not in self_species and species is not None:
                self_species[species] = []
            self_species[species].append(rec)
        logger.debug(f"Found {len(self_species)} species in alignment 1")

        # Group sequences by species in other alignment
        for rec in other[1:]:
            species,_,_,_ = self._parse_description(rec.description)
            if species not in other_species and species is not None:
                other_species[species] = []
            other_species[species].append(rec)
        logger.debug(f"Found {len(other_species)} species in alignment 2")

        # Merge records from both alignments
        merged = [SeqRecord(self[0].seq + other[0].seq, id='Query', name='Query', description=self[0].name+'+'+other[0].name)]
        common_species = set(self_species.keys()) & set(other_species.keys())
        logger.debug(f"Found {len(common_species)} common species")
        redundant_species = []
        ## First concatenate the sequences that are unique to each alignment
        logger.debug("Looping through common species")
        for species in common_species:
            if len(self_species[species]) == 1 and len(other_species[species]) == 1:
                rec1 = self_species[species][0]
                rec2 = other_species[species][0]
                # logger.debug(f'rec1.name: {rec1.name}, rec2.name: {rec2.name}, species: {species}')
                new_rec = SeqRecord(rec1.seq + rec2.seq, id=species, name=species, description=rec1.name+'+'+rec2.name)
                merged.append(new_rec)
            else:
                redundant_species.append(species)
        ## Then concatenate the sequences that are redundant in both alignments
        logger.debug(f"Found {len(redundant_species)} redundant species")
        for species in redundant_species:
            identity = 0
            q_arr = np.array(list(self.querySequence))
            for rec in self_species[species]:
                r_arr = np.array(list(rec))
                this_identity = np.sum(q_arr == r_arr)
                if this_identity > identity:
                    identity = this_identity
                    most_similar_seq_in_self = rec
            identity = 0
            q_arr = np.array(list(other.querySequence))
            for rec in other_species[species]:
                r_arr = np.array(list(rec))
                this_identity = np.sum(q_arr == r_arr)
                if this_identity > identity:
                    identity = this_identity
                    most_similar_seq_in_other = rec
            new_rec = SeqRecord(most_similar_seq_in_self.seq + most_similar_seq_in_other.seq, id=species, name=species, description=most_similar_seq_in_self.name+'+'+most_similar_seq_in_other.name)
            merged.append(new_rec)
        logger.info(f'Alignment merged: {len(merged)} sequences, {len(merged[0])} positions')
        return TextAlignment(merged)

    def Downsample_Randomly(self, to):
        if to >= self.sequenceCount:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.sequenceCount} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.sequenceCount), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)
        chosen_records = [self._records[i] for i in chosen_indices]
        super().__init__(chosen_records)
        self._update_info()

    def To_Zarr(self, filename=None):
        if not filename: filename = f'{self.alignmentName}'
        species_list = ['Query']
        species_index_map = [0]

        # Group sequences by species
        species = OrderedDict()
        for rec in self[1:]:
            OS, GN, PE, SV = self._parse_description(rec.description)
            if OS is None: continue
            if not species.get(OS): species[OS] = []
            species[OS].append(str(rec.seq))

        p_seq = 1
        tempstr = str(self[0].seq)
        for species, sequences in species.items():
            sequence_count = len(sequences)
            # for multiple records of the same species,
            # put the one with the highest identity to the query sequence at the first position
            if sequence_count != 1:
                identity = 0
                top_identity_index = 0
                q_arr = np.array(list(self.querySequence))
                for this_index, rec in enumerate(sequences):
                    r_arr = np.array(list(rec))
                    this_identity = np.sum(q_arr == r_arr)
                    if this_identity > identity:
                        identity = this_identity
                        top_identity_index = this_index
                sequences.insert(0, sequences.pop(top_identity_index))
            for seq in sequences: tempstr += seq
            species_index_map.append(p_seq)
            species_list.append(species)
            p_seq += sequence_count
        matrix = self.Get_Numpy_Array(mapped=True, string=tempstr)
        # Remove sequences with invalid characters from the alignment
        rows_to_keep = np.max(matrix, axis=1) <= 20
        matrix = matrix[rows_to_keep]
        z = zarr.create_array(store=filename,shape=(self.sequenceCount, self.sequenceLength), dtype='int8')
        metadata = {
            'species_list': species_list, 
            'species_index_map': species_index_map,
            'saved_columns': list(self.saved_columns)
        }
        with open(os.path.join(filename, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        z[:] = matrix


class ZarrAlignment(object):
    def __init__(self, filename=None):
        self.matrix = np.zeros((0, 0), dtype=np.int8)
        self.metadata = {}
        self.species_list_map = {}
        self.saved_columns = []

        if isinstance(filename, str):
            try:
                with open(os.path.join(filename, 'metadata.pkl'), 'rb') as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                logger.error(f"Metadata file not found in {filename}, {e}")
                raise FileNotFoundError(f"Metadata file not found in {filename}")
            try:
                self.matrix = zarr.load(filename)
            except Exception as e:
                logger.error(f"Error loading Zarr file {filename}, {e}")
                raise FileNotFoundError(f"Error loading Zarr file {filename}")
            self.species_list_map = {species: i for i, species in enumerate(self.metadata['species_list'])}
            self.saved_columns = self.metadata.get('saved_columns', [])

    def __repr__(self):
        return "<%s instance (%i records of length %i) at %x>" % (
            self.__class__,
            self.matrix.shape[0],
            self.matrix.shape[1],
            id(self),
        )

    def __str__(self):
        return f"Alignment from Zarr with {len(self.metadata['species_list'])} species"

    def __add__(self, other):
        common_species = set(self.metadata['species_list']) & set(other.metadata['species_list'])

        new_instance = ZarrAlignment()
        new_instance.matrix = np.zeros((len(common_species), self.matrix.shape[1] + other.matrix.shape[1]), dtype=np.int8)
        new_instance.metadata = {
            'species_list': list(common_species),
            'species_index_map': list(range(len(common_species)))
        }
        new_instance.species_list_map = {species: i for i, species in enumerate(new_instance.metadata['species_list'])}
        for i, species in enumerate(common_species):
            index1 = self.metadata['species_index_map'][self.species_list_map[species]]
            index2 = other.metadata['species_index_map'][other.species_list_map[species]]
            new_instance.matrix[i] = np.concatenate((self.matrix[index1], other.matrix[index2]))

        return new_instance

    def Downsample_Randomly(self, to):
        if to >= self.matrix.shape[0]:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.matrix.shape[0]} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.matrix.shape[0]), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)  # Ensure the first row is included
        self.matrix = self.matrix[chosen_indices]

    def To_Npy(self, filename=None):
        if not filename: filename = f'{self.metadata["species_list"][0]}-{self.metadata["species_list"][-1]}.npy'
        np.save(filename, self.matrix)

if __name__ == '__main__':
    FORMAT = '%(asctime)s %(levelname)s From %(name)s: %(message)s'
    logging.basicConfig(format=FORMAT, stream=sys.stderr, level=logging.DEBUG)

    # start_time = time.time()
    # alignment1 = TextAlignment("/Users/simon/research/EVcouplings/test_run/MSA/MYC_HUMAN_1-454_b0.2/align/MYC_HUMAN_1-454_b0.2.sto")
    # alignment1.Filtering_MSA_Gap(50, 50)
    # alignment2 = TextAlignment("/Users/simon/research/EVcouplings/test_run/MSA/S15A1_HUMAN_1-708_b0.2/align/S15A1_HUMAN_1-708_b0.2.sto")
    # alignment2.Filtering_MSA_Gap(50, 50)
    # logger.info(f"Time taken for reducing: {time.time() - start_time:.2f}s")
    # start_time = time.time()
    # merged_alignment = alignment1 + alignment2
    # matrix = merged_alignment.Get_Numpy_Array(mapped=True)
    # logger.info(f"Time taken for concatenating: {time.time() - start_time:.2f}s")
    
    # for d in os.listdir('/Users/simon/Dev/EVsnap/ecoli'):
    #     align = ZarrAlignment(f'/Users/simon/Dev/EVsnap/ecoli/{d}')

    align = TextAlignment("/Users/simon/research/EVcouplings/MSA_subset_evaluation/PDXH_ECOLI_1-218_b0.5.a2m")

    matrix = align.Get_Numpy_Array(mapped=True)
    


    align.Filtering_MSA_Gap(50, 50)
    align.Filtering_MSA_Identity(0.8)
    print(align.Get_Numpy_Array(mapped=True).shape)
    # align.Downsample_Randomly(10000)
    # SeqIO.write(align, "PF00028_10k.fasta", "fasta")
