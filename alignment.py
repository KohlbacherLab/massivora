import os
import re
import time
import logging
import sys
import compute
import zarr
import pickle
import subprocess
import shutil

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
            "-A", os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.sto'),
            "--tblout", os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.tblout'),
            "--domtblout", os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.domtblout'),
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
            self.__init__(os.path.join(search_args['working_dir'], f'{_id}_b{threshold}.sto'))

    def Get_Numpy_Array(self, mapped=False, string=None):
        if not string: string = ''.join([str(record.seq) for record in self])
        matrix = compute.getAlignmentInNumpy(string, self.sequenceLength, self.sequenceCount)
        # First change all the '.' to '-'
        matrix[matrix == b'.'] = b'-'
        if mapped:
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

    def Filtering_MSA_Invalid(self):
        # Remove sequences with invalid characters from the alignment
        matrix = self.Get_Numpy_Array()
        encoded_matrix = self.Get_Numpy_Array(mapped=True)
        valid_rows = np.where(np.max(encoded_matrix, axis=1) <= 20)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), valid_rows)
        matrix = matrix[valid_rows]
        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
        self._update_info()

    def Filtering_MSA_Gap(self, row_keep_percentage, col_keep_percentage):
        row_keep_percentage /= 100
        col_keep_percentage /= 100

        # Remove gaps if the input is a stockholm file
        columns_to_keep = np.array([char != '-' for char in self._records[0].seq])
        start_time = time.time()
        matrix = self.Get_Numpy_Array()

        # encoded_matrix = self.Get_Numpy_Array(mapped=True)
        logger.debug(f"Time taken for numpylizing: {time.time() - start_time:.2f}s")
        matrix = matrix[:, columns_to_keep]
        # encoded_matrix = encoded_matrix[:, columns_to_keep]

        # Calculate the ratio of '-' in each row and filter out rows with a ratio higher than row_keep_percentage
        logger.debug("Calculate rows to keep")
        row_gap_ratios = np.sum(matrix == b'-', axis=1) / matrix.shape[1]
        indices_to_keep = np.where(row_gap_ratios <= row_keep_percentage)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), indices_to_keep)
        matrix = matrix[indices_to_keep]
        logger.debug(f"Matrix shape after row filtering: {matrix.shape}")

        # Calculate the ratio of '-' in each column and filter out columns with a ratio higher than col_keep_percentage
        logger.debug("Calculate columns to keep")
        L = matrix.shape[0]
        col_gap_ratios = np.sum(matrix == b'-', axis=0) / L
        cols_to_keep = np.where(col_gap_ratios < col_keep_percentage)[0]
        self.saved_columns = cols_to_keep.tolist()
        matrix = matrix[:, cols_to_keep]
        matrix = np.char.upper(matrix)

        ############
        # The EVcouplings version: make the invalid columns lowercase and remove them in PLMC
        # col_indices_to_lowercase = np.where(col_gap_ratios >= col_keep_percentage)[0]
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

    def To_Zarr(self, filename=None, overwrite=False):
        if not filename: filename = f'{self.alignmentName}'
        if os.path.exists(filename) & os.path.exists(os.path.join(filename, 'c')):
            if overwrite:
                logger.warning(f"File {filename} already exists. Overwriting it.")
                shutil.rmtree(os.path.join(filename, 'c'), ignore_errors=True)
                os.remove(os.path.join(filename, 'zarr.json'))
                os.remove(os.path.join(filename, 'metadata.pkl'))
            else:
                logger.error(f"File {filename} already exists. Use overwrite=True to overwrite it.")
                return
        species_list = ['Query']
        species_index_map = [0]

        # Group sequences by species
        species = OrderedDict()
        for rec in self[1:]:
            OS, GN, PE, SV = self._parse_description(rec.description)
            if OS is None: continue
            if OS not in species.keys(): species[OS] = []
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
        # Ensure that Query is the first species in both alignments
        common_species = sorted(list(set(self.metadata['species_list'][1:]) & set(other.metadata['species_list'][1:])))
        common_species.insert(0, self.metadata['species_list'][0])

        new_instance = ZarrAlignment()
        new_instance.metadata = {
            'species_list': common_species,
            'species_index_map': list(range(len(common_species))),
            'saved_columns': self.saved_columns + other.saved_columns
        }
        new_instance.saved_columns = new_instance.metadata['saved_columns']
        new_instance.species_list_map = {species: i for i, species in enumerate(new_instance.metadata['species_list'])}

        # species_list_map[species] is the index of the species in the species_list
        # metadata['species_index_map'][species_list_map[species]] is the index of the species in the matrix
        indices1 = np.array([self.metadata['species_index_map'][self.species_list_map[species]] for species in common_species])
        indices2 = np.array([other.metadata['species_index_map'][other.species_list_map[species]] for species in common_species])
        new_instance.matrix = np.concatenate((self.matrix[indices1], other.matrix[indices2]), axis=1)

        return new_instance

    def Downsample_Randomly(self, to):
        if to >= self.matrix.shape[0]:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.matrix.shape[0]} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.matrix.shape[0]), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)  # Ensure the first row is included
        self.matrix = self.matrix[chosen_indices]

        # Update metadata to reflect the downsampled alignment
        kept_count = len(chosen_indices)
        new_species_list = ['Query']
        new_species_index_map = list(range(kept_count))

        for i in range(kept_count-1):
            new_species_list.append(self.metadata['species_list'][chosen_indices[i + 1]])

        self.metadata['species_list'] = new_species_list
        self.metadata['species_index_map'] = new_species_index_map
        self.species_list_map = {species: i for i, species in enumerate(new_species_list)}

    def To_Text(self, restore_gaps=True, filename=None):
        # Create reverse mapping from numbers back to characters
        valid_chars = 'acdefghiklmnpqrstvwyACDEFGHIKLMNPQRSTVWY-'
        values = np.arange(-20, 21, dtype=np.int8)
        reverse_mapping = np.full(128, ord('X'), dtype=np.uint8)
        for char, value in zip(valid_chars, values):
            reverse_mapping[value + 20] = ord(char)

        # Convert numeric matrix back to character matrix
        ascii_codes = reverse_mapping[self.matrix + 20]
        char_matrix = np.frombuffer(ascii_codes.tobytes(), dtype='S1').reshape(self.matrix.shape)
        if restore_gaps:
            diff = np.diff(self.saved_columns)
            decrease_indices = np.where(diff < 0)[0]
            monomer_starts = [0]
            for ending in decrease_indices:
                monomer_starts.append(ending + 1)
            monomer_matrices = []
            for i in range(len(monomer_starts)):
                start_idx = monomer_starts[i]
                end_idx = monomer_starts[i + 1] if i + 1 < len(monomer_starts) else len(self.saved_columns)
                monomer_saved_cols = self.saved_columns[start_idx:end_idx]
                monomer_char_matrix = char_matrix[:, start_idx:end_idx]
                monomer_original_length = max(monomer_saved_cols) + 1
                monomer_original_matrix = np.full((self.matrix.shape[0], monomer_original_length), b'-', dtype='S1')
                monomer_original_matrix[:, monomer_saved_cols] = monomer_char_matrix
                monomer_matrices.append(monomer_original_matrix)
            original_matrix = np.concatenate(monomer_matrices, axis=1)
        else:
            original_matrix = char_matrix.copy()
        original_matrix = original_matrix.astype('U1')

        # Convert matrix back to SeqRecord objects
        seq_records = []
        i = 0
        for species, starting_idx in zip(self.metadata['species_list'][:-1], self.metadata['species_index_map'][1:]):
            for _ in range(i, starting_idx):
                row = original_matrix[i]
                sequence_str = ''.join(row.tolist())
                seq_record = SeqRecord(
                    Seq(sequence_str),
                    id=str(i),
                    name=str(i) + ' OS=' + species,
                    description='OS=' + species
                )
                seq_records.append(seq_record)
                i += 1
        species = self.metadata['species_list'][-1]
        for _ in range(i, original_matrix.shape[0]):
            row = original_matrix[i]
            sequence_str = ''.join(row.tolist())
            seq_record = SeqRecord(
                Seq(sequence_str),
                id=str(i),
                name=str(i) + ' OS=' + species,
                description='OS=' + species
            )
            seq_records.append(seq_record)
            i += 1

        if filename:
            SeqIO.write(seq_records, filename, "fasta")

        return TextAlignment(seq_records)

    def To_Npy(self, filename=None):
        if filename:
            np.save(filename, self.matrix)
        return self.matrix

if __name__ == '__main__':
    FORMAT = '%(asctime)s %(levelname)s - From %(name)s: %(message)s'
    logging.basicConfig(format=FORMAT, stream=sys.stderr, level=logging.DEBUG)

    # start_time = time.time()
    # alignment1 = TextAlignment("/Users/simon/research/EVcouplings/residue_distance/ATPA_HUMAN_b0.2.sto")
    # alignment1.Filtering_MSA_Gap(50, 50)
    # # alignment1.To_Zarr(overwrite=True)
    # alignment2 = TextAlignment("/Users/simon/research/EVcouplings/residue_distance/ATPB_HUMAN_b0.2.sto")
    # alignment2.Filtering_MSA_Gap(50, 50)
    # # alignment2.To_Zarr(overwrite=True)
    # logger.info(f"Time taken for reducing: {time.time() - start_time:.2f}s")
    # start_time = time.time()
    # merged_alignment = alignment1 + alignment2
    # matrix = merged_alignment.Get_Numpy_Array(mapped=True)
    # logger.info(f"Time taken for concatenating: {time.time() - start_time:.2f}s")

    # for d in os.listdir('/Users/simon/Dev/EVsnap/ecoli'):
    #     align = ZarrAlignment(f'/Users/simon/Dev/EVsnap/ecoli/{d}')

    # align = TextAlignment("/Users/simon/research/EVcouplings/MSA_subset_evaluation/msa/PDXH_ECOLI_1-218_b0.5.sto")
    align = TextAlignment("/Users/simon/research/EVcouplings/MSA_subset_evaluation/PDXH_ECOLI_1-218_b0.5.a2m")
    align.Filtering_MSA_Invalid()
    print(len(align))
    align.Filtering_MSA_Gap(50, 50)
    print(align.Get_Numpy_Array().shape)
    matrix = align.Get_Numpy_Array(mapped=True)
    print(np.max(matrix))


    # align.Filtering_MSA_Gap(50, 50)
    # print(align.Get_Numpy_Array(mapped=True).shape)
    # align.To_Zarr(overwrite=True)

    # align.Downsample_Randomly(10000)
    # SeqIO.write(align, "PF00028_10k.fasta", "fasta")

